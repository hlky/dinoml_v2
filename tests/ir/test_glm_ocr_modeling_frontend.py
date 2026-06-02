from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from dinoml.models.glm_ocr import (
    GlmOcrConfig,
    GlmOcrForConditionalGeneration,
    GlmOcrForConditionalGenerationDecode,
    GlmOcrForConditionalGenerationDecodeSessionStaticCache,
    GlmOcrForConditionalGenerationDecodeStaticCache,
    GlmOcrForConditionalGenerationImagePrefill,
    GlmOcrForConditionalGenerationImagePrefillWithCache,
    GlmOcrTextConfig,
    GlmOcrVisionConfig,
    GlmOcrVisionModel,
    glm_ocr_config_from_transformers_dict,
    glm_ocr_patch_embed_linear_weight,
    glm_ocr_prepare_inputs_for_generation,
    glm_ocr_required_text_weight_names,
    glm_ocr_rope_index,
    glm_ocr_stitch_image_features,
    glm_ocr_text_rope_embeddings,
    glm_ocr_vision_position_ids,
    glm_ocr_vision_rope_embeddings,
    glm_ocr_weights_from_safetensors_file,
)
from dinoml.models.kv_cache import (
    StaticKvCacheSpec,
    empty_static_kv_cache,
    seed_static_kv_cache,
    static_kv_cache_input_specs,
    write_static_kv_cache_update,
)


def test_glm_ocr_official_config_preserves_explicit_projection_widths():
    config_path = Path(r"H:\dinoml_v2_agents\agents\plans\transformers\glm_ocr\_sources\zai-org_GLM-OCR_config.json")
    config = glm_ocr_config_from_transformers_dict(json.loads(config_path.read_text(encoding="utf-8")))

    assert config.text_config.hidden_size == 1536
    assert config.text_config.head_dim == 128
    assert config.text_config.q_proj_size == 2048
    assert config.text_config.kv_proj_size == 1024
    assert config.text_config.num_key_value_groups == 2
    assert config.text_config.rope_parameters["mrope_section"] == [16, 24, 24]
    assert config.vision_config.patch_dim == 1176
    assert config.vision_config.out_hidden_size == 1536


def test_glm_ocr_text_rope_uses_mrope_sections():
    config = GlmOcrTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=10,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=6,
        rope_parameters={"rope_type": "default", "mrope_section": [1, 1, 1], "partial_rotary_factor": 1.0, "rope_theta": 10000.0},
        dtype="float32",
    )
    position_ids = np.asarray(
        [
            [[1, 2]],
            [[10, 20]],
            [[100, 200]],
        ],
        dtype=np.int64,
    )

    cos, sin = glm_ocr_text_rope_embeddings(position_ids, config)

    inv = np.asarray([1.0, 10000.0 ** (-2.0 / 6.0), 10000.0 ** (-4.0 / 6.0)], dtype=np.float32)
    expected_freqs = np.asarray(
        [
            [
                [1 * inv[0], 10 * inv[1], 100 * inv[2]],
                [2 * inv[0], 20 * inv[1], 200 * inv[2]],
            ]
        ],
        dtype=np.float32,
    )
    expected = np.concatenate([expected_freqs, expected_freqs], axis=-1)
    np.testing.assert_allclose(cos, np.cos(expected), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(sin, np.sin(expected), rtol=1e-6, atol=1e-6)


def test_glm_ocr_rope_index_matches_source_grouping_for_image_tokens():
    input_ids = np.asarray([[11, 12, 59280, 59280, 13]], dtype=np.int64)
    mm_token_type_ids = np.asarray([[0, 0, 1, 1, 0]], dtype=np.int64)

    position_ids, deltas = glm_ocr_rope_index(
        input_ids,
        mm_token_type_ids,
        image_grid_thw=np.asarray([[1, 2, 4]], dtype=np.int64),
        spatial_merge_size=2,
    )

    expected = np.asarray(
        [
            [[0, 1, 2, 2, 4]],
            [[0, 1, 2, 2, 4]],
            [[0, 1, 2, 3, 4]],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(position_ids, expected)
    np.testing.assert_array_equal(deltas, np.asarray([[0]], dtype=np.int64))


def test_glm_ocr_vision_position_ids_preserve_merge_major_order():
    position_ids = glm_ocr_vision_position_ids(np.asarray([[1, 4, 4]], dtype=np.int64), spatial_merge_size=2)

    expected = np.asarray(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 0],
            [2, 1],
            [3, 0],
            [3, 1],
            [2, 2],
            [2, 3],
            [3, 2],
            [3, 3],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(position_ids, expected)


def test_glm_ocr_patch_embed_linear_weight_flattens_conv3d_weight():
    weight = np.arange(5 * 3 * 2 * 2 * 2, dtype=np.float32).reshape(5, 3, 2, 2, 2)
    linear = glm_ocr_patch_embed_linear_weight(weight)
    patches = np.arange(7 * 3 * 2 * 2 * 2, dtype=np.float32).reshape(7, -1)

    np.testing.assert_array_equal(linear, weight.reshape(5, -1))
    np.testing.assert_allclose(patches @ linear.T, patches @ weight.reshape(5, -1).T)


def test_glm_ocr_stitch_image_features_replaces_placeholders_in_row_major_order():
    input_ids = np.asarray([[1, 30, 2, 30]], dtype=np.int64)
    inputs_embeds = np.zeros((1, 4, 3), dtype=np.float32)
    image_features = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    stitched = glm_ocr_stitch_image_features(input_ids, inputs_embeds, image_features, image_token_id=30)

    np.testing.assert_array_equal(stitched[0, 1], image_features[0])
    np.testing.assert_array_equal(stitched[0, 3], image_features[1])
    np.testing.assert_array_equal(stitched[0, 0], np.zeros(3, dtype=np.float32))


def test_glm_ocr_generation_preparation_drops_visual_tensors_after_first_cached_step():
    prepared = glm_ocr_prepare_inputs_for_generation(
        {"input_ids": np.asarray([[1]]), "pixel_values": object(), "pixel_values_videos": object()},
        is_first_iteration=False,
        use_cache=True,
    )

    assert prepared["pixel_values"] is None
    assert prepared["pixel_values_videos"] is None


def test_glm_ocr_safetensors_loader_admits_required_tiny_weights(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    config = _tiny_config()
    weights = _tiny_weights(config)
    path = tmp_path / "model.safetensors"
    safetensors_torch.save_file({name: torch.from_numpy(value) for name, value in weights.items()}, path)

    loaded = glm_ocr_weights_from_safetensors_file(path, config, dtype="float32")

    assert set(loaded) == set(weights)
    np.testing.assert_array_equal(loaded["lm_head.weight"], weights["lm_head.weight"])


def test_glm_ocr_safetensors_loader_can_load_text_subset(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    config = _tiny_config()
    weights = _tiny_weights(config)
    path = tmp_path / "model.safetensors"
    safetensors_torch.save_file({name: torch.from_numpy(value) for name, value in weights.items()}, path)
    required = glm_ocr_required_text_weight_names(config)

    loaded = glm_ocr_weights_from_safetensors_file(path, config, dtype="float32", required_names=required)

    assert set(loaded) == set(required)
    assert "model.visual.patch_embed.proj.weight" not in loaded


def test_glm_ocr_tiny_text_trace_has_explicit_gqa_widths():
    config = _tiny_config()
    weights = _tiny_weights(config)
    model = GlmOcrForConditionalGeneration(config, weights)

    spec = dml.trace(
        model,
        inputs={
            "input_ids": dml.TensorSpec([1, 3], "int64"),
            "cos": dml.TensorSpec([1, 3, config.text_config.head_dim], "float32"),
            "sin": dml.TensorSpec([1, 3, config.text_config.head_dim], "float32"),
            "attention_mask": dml.TensorSpec([config.text_config.num_attention_heads, 3, 3], "float32"),
        },
        name="glm_ocr_tiny_text",
    )
    counts = Counter(node["op"] for node in spec.ir["nodes"])

    assert spec.ir["outputs"][0]["shape"] == [1, 3, config.text_config.vocab_size]
    assert counts["embedding"] == 1
    assert counts["t5_layer_norm"] == 5
    assert counts["bmm_rcr"] == 1
    assert counts["bmm_rrr"] == 1
    assert counts["gemm_rcr"] == 8


def test_glm_ocr_bf16_text_head_dim128_trace_uses_flash_attention():
    base_config = _tiny_config()
    text_config = replace(
        base_config.text_config,
        hidden_size=256,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [16, 24, 24],
            "partial_rotary_factor": 1.0,
            "rope_theta": 10000.0,
        },
        dtype="bfloat16",
        use_flash_attention_bias=True,
    )
    config = replace(base_config, text_config=text_config)
    weights = _tiny_weights(config)
    model = GlmOcrForConditionalGeneration(config, weights, logits_to_keep=1)

    spec = dml.trace(
        model,
        inputs={
            "input_ids": dml.TensorSpec([1, 3], "int64"),
            "cos": dml.TensorSpec([1, 3, config.text_config.head_dim], "bfloat16"),
            "sin": dml.TensorSpec([1, 3, config.text_config.head_dim], "bfloat16"),
            "attention_mask": dml.TensorSpec([config.text_config.num_attention_heads, 3, 3], "bfloat16"),
        },
        name="glm_ocr_bf16_text_head_dim128_flash",
    )
    counts = Counter(node["op"] for node in spec.ir["nodes"])

    assert counts["flash_attention_bias"] == config.text_config.num_hidden_layers
    assert counts["flash_attention"] == 0
    assert counts["bmm_rcr"] == 0
    assert counts["bmm_rrr"] == 0


def test_glm_ocr_tiny_text_logits_match_transformers_head():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers.models.glm_ocr.configuration_glm_ocr")
    from transformers.models.glm_ocr.configuration_glm_ocr import GlmOcrConfig as TransformersGlmOcrConfig
    from transformers.models.glm_ocr.modeling_glm_ocr import (
        GlmOcrForConditionalGeneration as TransformersGlmOcrForConditionalGeneration,
    )

    config = _tiny_config()
    weights = _tiny_weights(config)
    transformers_config = _tiny_transformers_config(config)
    transformers_model = TransformersGlmOcrForConditionalGeneration(transformers_config)
    state_dict = transformers_model.state_dict()
    state_dict.update({name: torch.from_numpy(value) for name, value in weights.items() if name in state_dict})
    transformers_model.load_state_dict(state_dict, strict=False)
    transformers_model.eval()
    input_ids = np.asarray([[1, config.image_token_id, 2]], dtype=np.int64)
    position_ids = np.repeat(np.arange(3, dtype=np.int64).reshape(1, 1, 3), 3, axis=0)
    cos, sin = glm_ocr_text_rope_embeddings(position_ids, config.text_config)
    attention_mask = np.triu(
        np.full((config.text_config.num_attention_heads, 3, 3), -1.0e4, dtype=np.float32),
        k=1,
    )
    spec = dml.trace(
        GlmOcrForConditionalGeneration(config, weights),
        inputs={
            "input_ids": dml.TensorSpec([1, 3], "int64"),
            "cos": dml.TensorSpec([1, 3, config.text_config.head_dim], "float32"),
            "sin": dml.TensorSpec([1, 3, config.text_config.head_dim], "float32"),
            "attention_mask": dml.TensorSpec([config.text_config.num_attention_heads, 3, 3], "float32"),
        },
        name="glm_ocr_tiny_text_parity",
    )

    actual = reference_numpy(
        spec,
        {
            "input_ids": input_ids,
            "cos": cos,
            "sin": sin,
            "attention_mask": attention_mask,
        },
    )["logits"]
    with torch.inference_mode():
        expected = transformers_model(
            input_ids=torch.from_numpy(input_ids),
            position_ids=torch.from_numpy(position_ids),
            logits_to_keep=0,
        ).logits.detach().cpu().numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_glm_ocr_tiny_decode_trace_appends_kv_cache_and_keeps_one_token_logits():
    config = _tiny_config()
    weights = _tiny_weights(config)
    model = GlmOcrForConditionalGenerationDecode(config, weights)
    past_len = 2
    total_len = past_len + 1

    inputs = {
        "input_ids": dml.TensorSpec([1, 1], "int64"),
        "cos": dml.TensorSpec([1, 1, config.text_config.head_dim], "float32"),
        "sin": dml.TensorSpec([1, 1, config.text_config.head_dim], "float32"),
        "attention_mask": dml.TensorSpec([config.text_config.num_attention_heads, 1, total_len], "float32"),
    }
    for layer_idx in range(config.text_config.num_hidden_layers):
        inputs[f"past_key_{layer_idx}"] = dml.TensorSpec(
            [1, config.text_config.num_key_value_heads, past_len, config.text_config.head_dim],
            "float32",
        )
        inputs[f"past_value_{layer_idx}"] = dml.TensorSpec(
            [1, config.text_config.num_key_value_heads, past_len, config.text_config.head_dim],
            "float32",
        )

    spec = dml.trace(model, inputs=inputs, name="glm_ocr_tiny_decode")
    output_shapes = {output["name"]: output["shape"] for output in spec.ir["outputs"]}
    counts = Counter(node["op"] for node in spec.ir["nodes"])

    assert output_shapes["logits"] == [1, 1, config.text_config.vocab_size]
    assert output_shapes["present_key_0"] == [1, config.text_config.num_key_value_heads, total_len, config.text_config.head_dim]
    assert output_shapes["present_value_0"] == [1, config.text_config.num_key_value_heads, total_len, config.text_config.head_dim]
    assert counts["concatenate"] == 2
    assert counts["bmm_rcr"] == 1
    assert counts["bmm_rrr"] == 1


def test_static_kv_cache_helpers_seed_and_update_standard_names():
    spec = StaticKvCacheSpec(
        num_layers=1,
        batch=1,
        num_key_value_heads=2,
        max_cache_len=5,
        head_dim=3,
        dtype="float32",
    )
    prefill_outputs = {
        "present_key_0": np.ones((1, 2, 2, 3), dtype=np.float32),
        "present_value_0": np.full((1, 2, 2, 3), 2.0, dtype=np.float32),
    }
    cache = seed_static_kv_cache(prefill_outputs, spec)

    np.testing.assert_array_equal(cache["past_key_0"][:, :, :2, :], prefill_outputs["present_key_0"])
    np.testing.assert_array_equal(cache["past_value_0"][:, :, :2, :], prefill_outputs["present_value_0"])
    np.testing.assert_array_equal(cache["past_key_0"][:, :, 2:, :], 0.0)

    updates = {
        "new_key_0": np.full((1, 2, 1, 3), 3.0, dtype=np.float32),
        "new_value_0": np.full((1, 2, 1, 3), 4.0, dtype=np.float32),
    }
    write_static_kv_cache_update(cache, updates, spec, position=2)

    np.testing.assert_array_equal(cache["past_key_0"][:, :, 2:3, :], updates["new_key_0"])
    np.testing.assert_array_equal(cache["past_value_0"][:, :, 2:3, :], updates["new_value_0"])


def test_static_kv_cache_helpers_encode_float_outputs_for_bfloat16_storage():
    spec = StaticKvCacheSpec(
        num_layers=1,
        batch=1,
        num_key_value_heads=1,
        max_cache_len=2,
        head_dim=2,
        dtype="bfloat16",
    )
    outputs = {
        "present_key_0": np.asarray([[[[-1.0, 0.5]]]], dtype=np.float32),
        "present_value_0": np.asarray([[[[2.0, -3.0]]]], dtype=np.float32),
    }

    cache = seed_static_kv_cache(outputs, spec)

    assert cache["past_key_0"].dtype == np.uint16
    assert cache["past_key_0"][0, 0, 0, 0] != np.uint16(0)
    assert cache["past_value_0"][0, 0, 0, 1] != np.uint16(0)


def test_glm_ocr_tiny_static_cache_decode_returns_one_token_cache_updates():
    config = _tiny_config()
    weights = _tiny_weights(config)
    model = GlmOcrForConditionalGenerationDecodeStaticCache(config, weights)
    max_past_len = 5
    cache_spec = StaticKvCacheSpec(
        num_layers=config.text_config.num_hidden_layers,
        batch=1,
        num_key_value_heads=config.text_config.num_key_value_heads,
        max_cache_len=max_past_len,
        head_dim=config.text_config.head_dim,
        dtype="float32",
    )
    inputs = {
        "input_ids": dml.TensorSpec([1, 1], "int64"),
        "cos": dml.TensorSpec([1, 1, config.text_config.head_dim], "float32"),
        "sin": dml.TensorSpec([1, 1, config.text_config.head_dim], "float32"),
        "attention_mask": dml.TensorSpec([config.text_config.num_attention_heads, 1, max_past_len + 1], "float32"),
        **static_kv_cache_input_specs(cache_spec),
    }

    spec = dml.trace(model, inputs=inputs, name="glm_ocr_tiny_decode_static_cache")
    output_shapes = {output["name"]: output["shape"] for output in spec.ir["outputs"]}
    counts = Counter(node["op"] for node in spec.ir["nodes"])

    assert output_shapes["logits"] == [1, 1, config.text_config.vocab_size]
    assert output_shapes["new_key_0"] == [1, config.text_config.num_key_value_heads, 1, config.text_config.head_dim]
    assert output_shapes["new_value_0"] == [1, config.text_config.num_key_value_heads, 1, config.text_config.head_dim]
    assert counts["concatenate"] == 2
    assert counts["bmm_rcr"] == 1
    assert counts["bmm_rrr"] == 1


@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
def test_glm_ocr_tiny_static_cache_decode_uses_flash_attention_cache_path(dtype: str):
    base_config = _tiny_config()
    config = replace(base_config, text_config=replace(base_config.text_config, dtype=dtype, use_flash_attention_bias=True))
    weights = _tiny_weights(config)
    model = GlmOcrForConditionalGenerationDecodeStaticCache(config, weights)
    max_past_len = 5
    cache_spec = StaticKvCacheSpec(
        num_layers=config.text_config.num_hidden_layers,
        batch=1,
        num_key_value_heads=config.text_config.num_key_value_heads,
        max_cache_len=max_past_len,
        head_dim=config.text_config.head_dim,
        dtype=dtype,
    )
    inputs = {
        "input_ids": dml.TensorSpec([1, 1], "int64"),
        "cos": dml.TensorSpec([1, 1, config.text_config.head_dim], dtype),
        "sin": dml.TensorSpec([1, 1, config.text_config.head_dim], dtype),
        "attention_mask": dml.TensorSpec([config.text_config.num_attention_heads, 1, max_past_len], dtype),
        "cache_seqlens": dml.TensorSpec([1], "int32"),
        **static_kv_cache_input_specs(cache_spec),
    }

    spec = dml.trace(model, inputs=inputs, name="glm_ocr_tiny_decode_static_cache_flash")
    counts = Counter(node["op"] for node in spec.ir["nodes"])

    assert counts["flash_attention_static_kv_cache_bias"] == 1
    assert counts["flash_attention_static_kv_cache"] == 0
    assert counts["concatenate"] == 0
    assert counts["bmm_rcr"] == 0
    assert counts["bmm_rrr"] == 0


def test_glm_ocr_tiny_session_static_cache_decode_uses_state_buffers():
    base_config = _tiny_config()
    config = replace(
        base_config,
        text_config=replace(base_config.text_config, dtype="bfloat16", use_flash_attention_bias=True),
    )
    weights = _tiny_weights(config)
    max_cache_len = 5
    model = GlmOcrForConditionalGenerationDecodeSessionStaticCache(
        config,
        weights,
        max_cache_len=max_cache_len,
    )
    inputs = {
        "input_ids": dml.TensorSpec([1, 1], "int64"),
        "cos": dml.TensorSpec([1, 1, config.text_config.head_dim], "bfloat16"),
        "sin": dml.TensorSpec([1, 1, config.text_config.head_dim], "bfloat16"),
        "attention_mask": dml.TensorSpec([config.text_config.num_attention_heads, 1, max_cache_len], "bfloat16"),
        "cache_seqlens": dml.TensorSpec([1], "int32"),
    }

    spec = dml.trace(model, inputs=inputs, name="glm_ocr_tiny_decode_session_static_cache")
    counts = Counter(node["op"] for node in spec.ir["nodes"])
    input_names = {item["name"] for item in spec.ir["inputs"]}
    output_names = {item["name"] for item in spec.ir["outputs"]}
    state_names = {item["name"] for item in spec.ir["states"]}

    assert output_names == {"logits"}
    assert "past_key_0" not in input_names
    assert "past_value_0" not in input_names
    assert state_names == {"past_key_0", "past_value_0"}
    assert counts["flash_attention_static_kv_cache_bias"] == 1
    assert counts["flash_attention_static_kv_cache"] == 0
    assert counts["concatenate"] == 0
    assert counts["bmm_rcr"] == 0
    assert counts["bmm_rrr"] == 0


def test_glm_ocr_tiny_prefill_can_keep_only_last_token_logits():
    config = _tiny_config()
    weights = _tiny_weights(config)
    model = GlmOcrForConditionalGeneration(config, weights, logits_to_keep=1)

    spec = dml.trace(
        model,
        inputs={
            "input_ids": dml.TensorSpec([1, 3], "int64"),
            "cos": dml.TensorSpec([1, 3, config.text_config.head_dim], "float32"),
            "sin": dml.TensorSpec([1, 3, config.text_config.head_dim], "float32"),
            "attention_mask": dml.TensorSpec([config.text_config.num_attention_heads, 3, 3], "float32"),
        },
        name="glm_ocr_tiny_text_last_token",
    )

    assert spec.ir["outputs"][0]["shape"] == [1, 1, config.text_config.vocab_size]


def test_glm_ocr_tiny_vision_trace_uses_linear_patch_embed_and_downsample_rewrites():
    config = _tiny_config()
    weights = _tiny_weights(config)
    model = GlmOcrVisionModel(config.vision_config, weights)

    spec = dml.trace(
        model,
        inputs={
            "pixel_values": dml.TensorSpec([4, config.vision_config.patch_dim], "float32"),
            "cos": dml.TensorSpec([4, config.vision_config.head_dim], "float32"),
            "sin": dml.TensorSpec([4, config.vision_config.head_dim], "float32"),
        },
        name="glm_ocr_tiny_vision",
    )
    counts = Counter(node["op"] for node in spec.ir["nodes"])
    output_shapes = {output["name"]: output["shape"] for output in spec.ir["outputs"]}

    assert output_shapes == {"last_hidden_state": [1, config.vision_config.out_hidden_size], "pooler_output": [1, config.vision_config.out_hidden_size]}
    assert counts["conv2d_bias"] == 0
    assert counts["gemm_rcr_bias"] >= 6
    assert counts["bmm_rcr"] == 1
    assert counts["bmm_rrr"] == 1


def test_glm_ocr_tiny_vision_pooler_matches_transformers():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers.models.glm_ocr.configuration_glm_ocr")
    from transformers.models.glm_ocr.modeling_glm_ocr import (
        GlmOcrForConditionalGeneration as TransformersGlmOcrForConditionalGeneration,
    )

    config = _tiny_config()
    weights = _tiny_weights(config)
    transformers_config = _tiny_transformers_config(config)
    transformers_config._attn_implementation = "eager"
    transformers_model = TransformersGlmOcrForConditionalGeneration(transformers_config)
    state_dict = transformers_model.state_dict()
    state_dict.update({name: torch.from_numpy(value) for name, value in weights.items() if name in state_dict})
    transformers_model.load_state_dict(state_dict, strict=False)
    transformers_model.eval()
    grid_thw = np.asarray([[1, 4, 4]], dtype=np.int64)
    pixel_values = np.arange(16 * config.vision_config.patch_dim, dtype=np.float32).reshape(16, -1) / 100.0
    vision_position_ids = glm_ocr_vision_position_ids(grid_thw, config.vision_config.spatial_merge_size)
    cos, sin = glm_ocr_vision_rope_embeddings(vision_position_ids, head_dim=config.vision_config.head_dim)
    spec = dml.trace(
        GlmOcrVisionModel(config.vision_config, weights),
        inputs={
            "pixel_values": dml.TensorSpec([16, config.vision_config.patch_dim], "float32"),
            "cos": dml.TensorSpec([16, config.vision_config.head_dim], "float32"),
            "sin": dml.TensorSpec([16, config.vision_config.head_dim], "float32"),
        },
        name="glm_ocr_tiny_vision_parity",
    )

    actual = reference_numpy(spec, {"pixel_values": pixel_values, "cos": cos, "sin": sin})["pooler_output"]
    with torch.inference_mode():
        expected = transformers_model.model.get_image_features(
            pixel_values=torch.from_numpy(pixel_values),
            image_grid_thw=torch.from_numpy(grid_thw),
            return_dict=True,
        ).pooler_output[0].detach().cpu().numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_glm_ocr_bfloat16_vision_rope_computes_rotation_in_float32():
    config = _tiny_config()
    config = replace(config, vision_config=replace(config.vision_config, dtype="bfloat16"))
    weights = _tiny_weights(config)
    spec = dml.trace(
        GlmOcrVisionModel(config.vision_config, weights),
        inputs={
            "pixel_values": dml.TensorSpec([16, config.vision_config.patch_dim], "bfloat16"),
            "cos": dml.TensorSpec([16, config.vision_config.head_dim], "float32"),
            "sin": dml.TensorSpec([16, config.vision_config.head_dim], "float32"),
        },
        name="glm_ocr_tiny_vision_rope_bfloat16_float32_rotation",
    )

    cast_dtypes = [node.get("attrs", {}).get("dtype") for node in spec.ir["nodes"] if node["op"] == "cast"]

    assert cast_dtypes.count("float32") >= 2
    assert cast_dtypes.count("bfloat16") >= 2


def test_glm_ocr_tiny_image_prefill_logits_match_transformers():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers.models.glm_ocr.configuration_glm_ocr")
    from transformers.models.glm_ocr.modeling_glm_ocr import (
        GlmOcrForConditionalGeneration as TransformersGlmOcrForConditionalGeneration,
    )

    config = _tiny_config()
    weights = _tiny_weights(config)
    transformers_config = _tiny_transformers_config(config)
    transformers_config._attn_implementation = "eager"
    transformers_model = TransformersGlmOcrForConditionalGeneration(transformers_config)
    state_dict = transformers_model.state_dict()
    state_dict.update({name: torch.from_numpy(value) for name, value in weights.items() if name in state_dict})
    transformers_model.load_state_dict(state_dict, strict=False)
    transformers_model.eval()
    input_ids = np.asarray([[1, config.image_token_id, config.image_token_id, config.image_token_id, config.image_token_id, 2]], dtype=np.int64)
    mm_token_type_ids = np.asarray([[0, 1, 1, 1, 1, 0]], dtype=np.int64)
    image_grid_thw = np.asarray([[1, 4, 4]], dtype=np.int64)
    pixel_values = np.arange(16 * config.vision_config.patch_dim, dtype=np.float32).reshape(16, -1) / 100.0
    text_position_ids, _ = glm_ocr_rope_index(
        input_ids,
        mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=config.vision_config.spatial_merge_size,
    )
    text_cos, text_sin = glm_ocr_text_rope_embeddings(text_position_ids, config.text_config)
    vision_position_ids = glm_ocr_vision_position_ids(image_grid_thw, config.vision_config.spatial_merge_size)
    vision_cos, vision_sin = glm_ocr_vision_rope_embeddings(vision_position_ids, head_dim=config.vision_config.head_dim)
    seq_len = input_ids.shape[1]
    attention_mask = np.triu(
        np.full((config.text_config.num_attention_heads, seq_len, seq_len), -1.0e4, dtype=np.float32),
        k=1,
    )
    spec = dml.trace(
        GlmOcrForConditionalGenerationImagePrefill(config, weights, image_token_start=1),
        inputs={
            "input_ids": dml.TensorSpec([1, seq_len], "int64"),
            "pixel_values": dml.TensorSpec([16, config.vision_config.patch_dim], "float32"),
            "vision_cos": dml.TensorSpec([16, config.vision_config.head_dim], "float32"),
            "vision_sin": dml.TensorSpec([16, config.vision_config.head_dim], "float32"),
            "text_cos": dml.TensorSpec([1, seq_len, config.text_config.head_dim], "float32"),
            "text_sin": dml.TensorSpec([1, seq_len, config.text_config.head_dim], "float32"),
            "attention_mask": dml.TensorSpec([config.text_config.num_attention_heads, seq_len, seq_len], "float32"),
        },
        name="glm_ocr_tiny_image_prefill_parity",
    )

    actual = reference_numpy(
        spec,
        {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "vision_cos": vision_cos,
            "vision_sin": vision_sin,
            "text_cos": text_cos,
            "text_sin": text_sin,
            "attention_mask": attention_mask,
        },
    )["logits"]
    with torch.inference_mode():
        expected = transformers_model(
            input_ids=torch.from_numpy(input_ids),
            pixel_values=torch.from_numpy(pixel_values),
            image_grid_thw=torch.from_numpy(image_grid_thw),
            mm_token_type_ids=torch.from_numpy(mm_token_type_ids),
            position_ids=torch.from_numpy(text_position_ids),
            logits_to_keep=0,
        ).logits.detach().cpu().numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_glm_ocr_tiny_image_prefill_with_cache_outputs_decode_cache_shapes():
    config = _tiny_config()
    weights = _tiny_weights(config)
    input_ids = np.asarray([[1, config.image_token_id, config.image_token_id, config.image_token_id, config.image_token_id, 2]], dtype=np.int64)
    mm_token_type_ids = np.asarray([[0, 1, 1, 1, 1, 0]], dtype=np.int64)
    image_grid_thw = np.asarray([[1, 4, 4]], dtype=np.int64)
    pixel_values = np.arange(16 * config.vision_config.patch_dim, dtype=np.float32).reshape(16, -1) / 100.0
    text_position_ids, _ = glm_ocr_rope_index(
        input_ids,
        mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=config.vision_config.spatial_merge_size,
    )
    text_cos, text_sin = glm_ocr_text_rope_embeddings(text_position_ids, config.text_config)
    vision_position_ids = glm_ocr_vision_position_ids(image_grid_thw, config.vision_config.spatial_merge_size)
    vision_cos, vision_sin = glm_ocr_vision_rope_embeddings(vision_position_ids, head_dim=config.vision_config.head_dim)
    seq_len = input_ids.shape[1]
    attention_mask = np.triu(
        np.full((config.text_config.num_attention_heads, seq_len, seq_len), -1.0e4, dtype=np.float32),
        k=1,
    )
    spec = dml.trace(
        GlmOcrForConditionalGenerationImagePrefillWithCache(config, weights, image_token_start=1),
        inputs={
            "input_ids": dml.TensorSpec([1, seq_len], "int64"),
            "pixel_values": dml.TensorSpec([16, config.vision_config.patch_dim], "float32"),
            "vision_cos": dml.TensorSpec([16, config.vision_config.head_dim], "float32"),
            "vision_sin": dml.TensorSpec([16, config.vision_config.head_dim], "float32"),
            "text_cos": dml.TensorSpec([1, seq_len, config.text_config.head_dim], "float32"),
            "text_sin": dml.TensorSpec([1, seq_len, config.text_config.head_dim], "float32"),
            "attention_mask": dml.TensorSpec([config.text_config.num_attention_heads, seq_len, seq_len], "float32"),
        },
        name="glm_ocr_tiny_image_prefill_with_cache",
    )
    output_shapes = {output["name"]: output["shape"] for output in spec.ir["outputs"]}

    assert output_shapes["logits"] == [1, 1, config.text_config.vocab_size]
    for layer_idx in range(config.text_config.num_hidden_layers):
        assert output_shapes[f"present_key_{layer_idx}"] == [
            1,
            config.text_config.num_key_value_heads,
            seq_len,
            config.text_config.head_dim,
        ]
        assert output_shapes[f"present_value_{layer_idx}"] == [
            1,
            config.text_config.num_key_value_heads,
            seq_len,
            config.text_config.head_dim,
        ]


def _tiny_config() -> GlmOcrConfig:
    text = GlmOcrTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=10,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=6,
        rope_parameters={"rope_type": "default", "mrope_section": [1, 1, 1], "partial_rotary_factor": 1.0, "rope_theta": 10000.0},
        dtype="float32",
    )
    vision = GlmOcrVisionConfig(
        depth=1,
        hidden_size=8,
        intermediate_size=12,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=8,
        dtype="float32",
    )
    return GlmOcrConfig(text_config=text, vision_config=vision, image_token_id=30, video_token_id=31)


def _tiny_transformers_config(config: GlmOcrConfig):
    from transformers.models.glm_ocr.configuration_glm_ocr import GlmOcrConfig as TransformersGlmOcrConfig

    return TransformersGlmOcrConfig(
        text_config={
            "vocab_size": config.text_config.vocab_size,
            "hidden_size": config.text_config.hidden_size,
            "intermediate_size": config.text_config.intermediate_size,
            "num_hidden_layers": config.text_config.num_hidden_layers,
            "num_attention_heads": config.text_config.num_attention_heads,
            "num_key_value_heads": config.text_config.num_key_value_heads,
            "head_dim": config.text_config.head_dim,
            "hidden_act": "silu",
            "rms_norm_eps": config.text_config.rms_norm_eps,
            "attention_dropout": 0.0,
            "rope_parameters": dict(config.text_config.rope_parameters),
            "pad_token_id": 0,
        },
        vision_config={
            "depth": config.vision_config.depth,
            "hidden_size": config.vision_config.hidden_size,
            "intermediate_size": config.vision_config.intermediate_size,
            "num_heads": config.vision_config.num_heads,
            "in_channels": config.vision_config.in_channels,
            "patch_size": config.vision_config.patch_size,
            "temporal_patch_size": config.vision_config.temporal_patch_size,
            "spatial_merge_size": config.vision_config.spatial_merge_size,
            "out_hidden_size": config.vision_config.out_hidden_size,
            "hidden_act": "silu",
            "attention_bias": True,
            "rms_norm_eps": config.vision_config.rms_norm_eps,
        },
        image_token_id=config.image_token_id,
        video_token_id=config.video_token_id,
    )


def _tiny_weights(config: GlmOcrConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(123)
    weights: dict[str, np.ndarray] = {}

    def add(name: str, shape: tuple[int, ...]) -> None:
        weights[name] = rng.normal(0.0, 0.02, shape).astype(np.float32)

    text = config.text_config
    add("model.language_model.embed_tokens.weight", (text.vocab_size, text.hidden_size))
    add("model.language_model.norm.weight", (text.hidden_size,))
    add("lm_head.weight", (text.vocab_size, text.hidden_size))
    for idx in range(text.num_hidden_layers):
        prefix = f"model.language_model.layers.{idx}"
        add(f"{prefix}.input_layernorm.weight", (text.hidden_size,))
        add(f"{prefix}.self_attn.q_proj.weight", (text.q_proj_size, text.hidden_size))
        add(f"{prefix}.self_attn.k_proj.weight", (text.kv_proj_size, text.hidden_size))
        add(f"{prefix}.self_attn.v_proj.weight", (text.kv_proj_size, text.hidden_size))
        add(f"{prefix}.self_attn.o_proj.weight", (text.hidden_size, text.q_proj_size))
        add(f"{prefix}.post_self_attn_layernorm.weight", (text.hidden_size,))
        add(f"{prefix}.post_attention_layernorm.weight", (text.hidden_size,))
        add(f"{prefix}.mlp.gate_up_proj.weight", (text.intermediate_size * 2, text.hidden_size))
        add(f"{prefix}.mlp.down_proj.weight", (text.hidden_size, text.intermediate_size))
        add(f"{prefix}.post_mlp_layernorm.weight", (text.hidden_size,))

    vision = config.vision_config
    add("model.visual.patch_embed.proj.weight", (vision.hidden_size, vision.in_channels, vision.temporal_patch_size, vision.patch_size, vision.patch_size))
    add("model.visual.patch_embed.proj.bias", (vision.hidden_size,))
    add("model.visual.post_layernorm.weight", (vision.hidden_size,))
    add("model.visual.downsample.weight", (vision.out_hidden_size, vision.hidden_size, vision.spatial_merge_size, vision.spatial_merge_size))
    add("model.visual.downsample.bias", (vision.out_hidden_size,))
    add("model.visual.merger.proj.weight", (vision.out_hidden_size, vision.out_hidden_size))
    add("model.visual.merger.post_projection_norm.weight", (vision.out_hidden_size,))
    add("model.visual.merger.post_projection_norm.bias", (vision.out_hidden_size,))
    add("model.visual.merger.gate_proj.weight", (vision.out_hidden_size * vision.in_channels, vision.out_hidden_size))
    add("model.visual.merger.up_proj.weight", (vision.out_hidden_size * vision.in_channels, vision.out_hidden_size))
    add("model.visual.merger.down_proj.weight", (vision.out_hidden_size, vision.out_hidden_size * vision.in_channels))
    for idx in range(vision.depth):
        prefix = f"model.visual.blocks.{idx}"
        add(f"{prefix}.norm1.weight", (vision.hidden_size,))
        add(f"{prefix}.attn.qkv.weight", (vision.hidden_size * 3, vision.hidden_size))
        add(f"{prefix}.attn.qkv.bias", (vision.hidden_size * 3,))
        add(f"{prefix}.attn.proj.weight", (vision.hidden_size, vision.hidden_size))
        add(f"{prefix}.attn.proj.bias", (vision.hidden_size,))
        add(f"{prefix}.attn.q_norm.weight", (vision.head_dim,))
        add(f"{prefix}.attn.k_norm.weight", (vision.head_dim,))
        add(f"{prefix}.norm2.weight", (vision.hidden_size,))
        add(f"{prefix}.mlp.gate_proj.weight", (vision.intermediate_size, vision.hidden_size))
        add(f"{prefix}.mlp.gate_proj.bias", (vision.intermediate_size,))
        add(f"{prefix}.mlp.up_proj.weight", (vision.intermediate_size, vision.hidden_size))
        add(f"{prefix}.mlp.up_proj.bias", (vision.intermediate_size,))
        add(f"{prefix}.mlp.down_proj.weight", (vision.hidden_size, vision.intermediate_size))
        add(f"{prefix}.mlp.down_proj.bias", (vision.hidden_size,))
    return weights
