from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

import dinoml as dml
from dinoml.compiler import _validate_mvp_runtime_contract
from dinoml.models.kv_cache import StaticKvCacheSpec, static_kv_cache_input_specs
from dinoml.passes.core import shape_type_infer
from dinoml.passes.validation import validate_ir
from dinoml.reference import reference_numpy
from dinoml.shapes import Dim, symbolic_int_expr
from dinoml.models.qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGenerationDecode,
    Qwen2_5_VLForConditionalGenerationDecodeStaticCache,
    Qwen2_5_VLForConditionalGenerationImagePrefillWithCache,
    Qwen2_5_VLTextConfig,
    Qwen2_5_VLVisionConfig,
    Qwen2_5_VLVisionModel,
    qwen2_5_vl_rope_index,
    qwen2_5_vl_text_rope_embeddings,
    qwen2_5_vl_vision_position_ids,
    qwen2_5_vl_vision_rope_embeddings,
)


def test_qwen2_5_vl_tiny_vision_trace_uses_packed_patch_gemm_and_merger():
    config = _tiny_config()
    weights = _tiny_weights(config)

    spec = dml.trace(
        Qwen2_5_VLVisionModel(config.vision_config, weights, grid_thw=None),
        inputs={
            "pixel_values": dml.TensorSpec([4, config.vision_config.patch_dim], "float32"),
            "cos": dml.TensorSpec([4, config.vision_config.head_dim], "float32"),
            "sin": dml.TensorSpec([4, config.vision_config.head_dim], "float32"),
        },
        name="qwen2_5_vl_tiny_vision",
    )
    counts = Counter(node["op"] for node in spec.ir["nodes"])
    output_shapes = {output["name"]: output["shape"] for output in spec.ir["outputs"]}

    assert output_shapes == {"pooler_output": [1, config.vision_config.out_hidden_size]}
    assert counts["glm_ocr_vision_rope"] == config.vision_config.depth
    assert counts["conv2d_bias"] == 0
    assert counts["conv3d"] == 0
    assert counts["bmm_rcr"] == 1
    assert counts["bmm_rrr"] == 1


def test_qwen2_5_vl_vision_rejects_baked_grid_thw():
    config = _tiny_config()
    weights = _tiny_weights(config)
    grid_thw = np.asarray([[1, 2, 2]], dtype=np.int64)

    with pytest.raises(ValueError, match="grid_thw must remain external"):
        Qwen2_5_VLVisionModel(config.vision_config, weights, grid_thw=grid_thw)


def test_qwen2_5_vl_tiny_vision_trace_preserves_symbolic_patch_count():
    config = _tiny_config()
    weights = _tiny_weights(config)
    patch_count = Dim("patch_count", min=4, max=16, divisible_by=4, typical=4, buckets=(4, 8, 16))

    spec = dml.trace(
        Qwen2_5_VLVisionModel(config.vision_config, weights, grid_thw=None),
        inputs={
            "pixel_values": dml.TensorSpec([patch_count, config.vision_config.patch_dim], "float32"),
            "cos": dml.TensorSpec([patch_count, config.vision_config.head_dim], "float32"),
            "sin": dml.TensorSpec([patch_count, config.vision_config.head_dim], "float32"),
        },
        name="qwen2_5_vl_tiny_vision_dynamic",
    )
    validate_ir(shape_type_infer(deepcopy(spec.ir)))
    counts = Counter(node["op"] for node in spec.ir["nodes"])
    output_shape = spec.ir["outputs"][0]["shape"]
    output_shape_spec = spec.ir["outputs"][0]["shape_spec"]

    assert counts["glm_ocr_vision_rope"] == config.vision_config.depth
    assert output_shape == [4, config.vision_config.out_hidden_size]
    assert output_shape_spec[0] == {
        "kind": "int_expr",
        "op": "div",
        "lhs": {
            "kind": "dim",
            "name": "patch_count",
            "min": 4,
            "max": 16,
            "divisible_by": 4,
            "typical": 4,
            "buckets": [4, 8, 16],
        },
        "rhs": 4,
    }
    assert output_shape_spec[1] == config.vision_config.out_hidden_size


def test_qwen2_5_vl_dynamic_bfloat16_vision_uses_varlen_window_attention():
    base_config = _tiny_config()
    config = replace(
        base_config,
        vision_config=replace(
            base_config.vision_config,
            depth=2,
            dtype="bfloat16",
            use_flash_attention=True,
            fullatt_block_indexes=(1,),
        ),
    )
    weights = _tiny_weights(config)
    patch_count = Dim("patch_count", min=4, max=16, divisible_by=4, typical=4, buckets=(4, 8, 16))
    image_feature_count = Dim("image_feature_count", min=1, max=4, divisible_by=1, typical=1, buckets=(1, 2, 4))
    window_cu_count = Dim("vision_window_cu_count", min=2, max=5, divisible_by=1, typical=2, buckets=(2, 5))

    spec = dml.trace(
        Qwen2_5_VLVisionModel(
            config.vision_config,
            weights,
            grid_thw=None,
            max_full_seqlen=16,
            max_window_seqlen=4,
        ),
        inputs={
            "pixel_values": dml.TensorSpec([patch_count, config.vision_config.patch_dim], "bfloat16"),
            "cos": dml.TensorSpec([patch_count, config.vision_config.head_dim], "float32"),
            "sin": dml.TensorSpec([patch_count, config.vision_config.head_dim], "float32"),
            "full_cu_seqlens": dml.TensorSpec([2], "int32"),
            "window_cu_seqlens": dml.TensorSpec([window_cu_count], "int32"),
            "reverse_window_index": dml.TensorSpec([image_feature_count], "int32"),
        },
        name="qwen2_5_vl_dynamic_bfloat16_vision_varlen",
    )
    validate_ir(shape_type_infer(deepcopy(spec.ir)))
    counts = Counter(node["op"] for node in spec.ir["nodes"])

    assert counts["flash_attention_varlen"] == 2
    assert counts["flash_attention"] == 0
    assert counts["runtime_index_select"] == 1
    assert spec.ir["outputs"][0]["shape_spec"][0] == {
        "kind": "dim",
        "name": "image_feature_count",
        "min": 1,
        "max": 4,
        "divisible_by": 1,
        "typical": 1,
        "buckets": [1, 2, 4],
    }


def test_qwen2_5_vl_tiny_image_prefill_outputs_decode_cache_shapes():
    config = _tiny_config()
    weights = _tiny_weights(config)
    fixture = _prefill_fixture(config)
    seq_len = fixture["input_ids"].shape[1]
    patch_count = fixture["pixel_values"].shape[0]

    spec = dml.trace(
        Qwen2_5_VLForConditionalGenerationImagePrefillWithCache(config, weights, grid_thw=None),
        inputs={
            "input_ids": dml.TensorSpec([1, seq_len], "int64"),
            "pixel_values": dml.TensorSpec([patch_count, config.vision_config.patch_dim], "float32"),
            "vision_cos": dml.TensorSpec([patch_count, config.vision_config.head_dim], "float32"),
            "vision_sin": dml.TensorSpec([patch_count, config.vision_config.head_dim], "float32"),
            "text_cos": dml.TensorSpec([1, seq_len, config.text_config.head_dim], "float32"),
            "text_sin": dml.TensorSpec([1, seq_len, config.text_config.head_dim], "float32"),
            "attention_mask": dml.TensorSpec([config.text_config.num_attention_heads, seq_len, seq_len], "float32"),
            "image_grid_thw": dml.TensorSpec([1, 3], "int64"),
        },
        name="qwen2_5_vl_tiny_image_prefill_with_cache",
    )
    validate_ir(shape_type_infer(deepcopy(spec.ir)))
    counts = Counter(node["op"] for node in spec.ir["nodes"])
    input_names = {input_info["name"] for input_info in spec.ir["inputs"]}
    output_shapes = {output["name"]: output["shape"] for output in spec.ir["outputs"]}

    assert counts["qwen2_5_vl_stitch_image_features"] == 1
    assert "image_grid_thw" in input_names
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


def test_qwen2_5_vl_tiny_image_prefill_preserves_dynamic_sequence_and_patch_shapes():
    config = _tiny_config()
    weights = _tiny_weights(config)
    patch_count = Dim("patch_count", min=4, max=16, divisible_by=4, typical=4, buckets=(4, 8, 16))
    seq_len = Dim("seq_len", min=3, max=8, divisible_by=1, typical=3, buckets=(3, 6, 8))

    spec = dml.trace(
        Qwen2_5_VLForConditionalGenerationImagePrefillWithCache(config, weights, grid_thw=None),
        inputs={
            "input_ids": dml.TensorSpec([1, seq_len], "int64"),
            "pixel_values": dml.TensorSpec([patch_count, config.vision_config.patch_dim], "float32"),
            "vision_cos": dml.TensorSpec([patch_count, config.vision_config.head_dim], "float32"),
            "vision_sin": dml.TensorSpec([patch_count, config.vision_config.head_dim], "float32"),
            "text_cos": dml.TensorSpec([1, seq_len, config.text_config.head_dim], "float32"),
            "text_sin": dml.TensorSpec([1, seq_len, config.text_config.head_dim], "float32"),
            "attention_mask": dml.TensorSpec([config.text_config.num_attention_heads, seq_len, seq_len], "float32"),
            "image_grid_thw": dml.TensorSpec([1, 3], "int64"),
        },
        name="qwen2_5_vl_tiny_image_prefill_dynamic",
    )
    validate_ir(shape_type_infer(deepcopy(spec.ir)))
    counts = Counter(node["op"] for node in spec.ir["nodes"])
    outputs = {output["name"]: output for output in spec.ir["outputs"]}

    assert counts["qwen2_5_vl_stitch_image_features"] == 1
    assert outputs["logits"]["shape"] == [1, 1, config.text_config.vocab_size]
    assert outputs["present_key_0"]["shape"] == [1, config.text_config.num_key_value_heads, 8, config.text_config.head_dim]
    assert outputs["present_key_0"]["shape_spec"][2] == {
        "kind": "dim",
        "name": "seq_len",
        "min": 3,
        "max": 8,
        "divisible_by": 1,
        "typical": 3,
        "buckets": [3, 6, 8],
    }


def test_qwen2_5_vl_dense_prefill_requires_explicit_causal_attention_mask():
    config = _tiny_config()
    weights = _tiny_weights(config)
    fixture = _prefill_fixture(config)

    with pytest.raises(ValueError, match="dense text prefill requires an explicit causal attention_mask"):
        dml.trace(
            Qwen2_5_VLForConditionalGenerationImagePrefillWithCache(config, weights, grid_thw=None),
            inputs={
                "input_ids": dml.TensorSpec(list(fixture["input_ids"].shape), "int64"),
                "pixel_values": dml.TensorSpec(list(fixture["pixel_values"].shape), "float32"),
                "vision_cos": dml.TensorSpec(list(fixture["vision_cos"].shape), "float32"),
                "vision_sin": dml.TensorSpec(list(fixture["vision_sin"].shape), "float32"),
                "text_cos": dml.TensorSpec(list(fixture["text_cos"].shape), "float32"),
                "text_sin": dml.TensorSpec(list(fixture["text_sin"].shape), "float32"),
                "image_grid_thw": dml.TensorSpec(list(fixture["image_grid_thw"].shape), "int64"),
            },
            name="qwen2_5_vl_dense_prefill_without_mask",
        )


def test_qwen2_5_vl_tiny_decode_appends_gqa_kv_cache():
    config = _tiny_config()
    weights = _tiny_weights(config)
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

    spec = dml.trace(Qwen2_5_VLForConditionalGenerationDecode(config, weights), inputs=inputs, name="qwen2_5_vl_tiny_decode")
    validate_ir(shape_type_infer(deepcopy(spec.ir)))
    output_shapes = {output["name"]: output["shape"] for output in spec.ir["outputs"]}
    counts = Counter(node["op"] for node in spec.ir["nodes"])

    assert output_shapes["logits"] == [1, 1, config.text_config.vocab_size]
    assert output_shapes["present_key_0"] == [1, config.text_config.num_key_value_heads, total_len, config.text_config.head_dim]
    assert output_shapes["present_value_0"] == [1, config.text_config.num_key_value_heads, total_len, config.text_config.head_dim]
    assert _cache_concatenate_count(spec) == 2
    assert counts["bmm_rcr"] == 1
    assert counts["bmm_rrr"] == 1


def test_qwen2_5_vl_dense_cached_decode_requires_attention_mask_for_multi_token_queries():
    config = _tiny_config()
    weights = _tiny_weights(config)
    past_len = 2
    query_len = 2
    inputs = {
        "input_ids": dml.TensorSpec([1, query_len], "int64"),
        "cos": dml.TensorSpec([1, query_len, config.text_config.head_dim], "float32"),
        "sin": dml.TensorSpec([1, query_len, config.text_config.head_dim], "float32"),
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

    with pytest.raises(ValueError, match="dense cached text attention requires an explicit attention_mask when query length != 1"):
        dml.trace(
            Qwen2_5_VLForConditionalGenerationDecode(config, weights),
            inputs=inputs,
            name="qwen2_5_vl_dense_decode_without_mask",
        )


def test_qwen2_5_vl_tiny_decode_preserves_dynamic_past_len_with_flash_bias():
    base_config = _tiny_config()
    config = replace(
        base_config,
        text_config=replace(base_config.text_config, dtype="bfloat16", use_flash_attention=True, use_flash_attention_bias=True),
    )
    weights = _tiny_weights(config)
    past_len = Dim("past_len", min=2, max=5, divisible_by=1, typical=2, buckets=(2, 5))
    total_len = symbolic_int_expr("add", past_len.to_json(), 1)
    inputs = {
        "input_ids": dml.TensorSpec([1, 1], "int64"),
        "cos": dml.TensorSpec([1, 1, config.text_config.head_dim], "bfloat16"),
        "sin": dml.TensorSpec([1, 1, config.text_config.head_dim], "bfloat16"),
        "attention_mask": dml.TensorSpec([config.text_config.num_attention_heads, 1, total_len], "bfloat16"),
    }
    for layer_idx in range(config.text_config.num_hidden_layers):
        inputs[f"past_key_{layer_idx}"] = dml.TensorSpec(
            [1, config.text_config.num_key_value_heads, past_len, config.text_config.head_dim],
            "bfloat16",
        )
        inputs[f"past_value_{layer_idx}"] = dml.TensorSpec(
            [1, config.text_config.num_key_value_heads, past_len, config.text_config.head_dim],
            "bfloat16",
        )

    spec = dml.trace(Qwen2_5_VLForConditionalGenerationDecode(config, weights), inputs=inputs, name="qwen2_5_vl_tiny_decode_dynamic")
    validate_ir(shape_type_infer(deepcopy(spec.ir)))
    counts = Counter(node["op"] for node in spec.ir["nodes"])
    output = next(output for output in spec.ir["outputs"] if output["name"] == "present_key_0")

    assert counts["flash_attention_bias"] == 1
    assert counts["bmm_rcr"] == 0
    assert counts["bmm_rrr"] == 0
    assert output["shape"] == [1, config.text_config.num_key_value_heads, 6, config.text_config.head_dim]
    assert output["shape_spec"][2] == total_len


def test_qwen2_5_vl_tiny_static_cache_decode_returns_one_token_updates():
    config = _tiny_config()
    weights = _tiny_weights(config)
    max_cache_len = 5
    cache_spec = StaticKvCacheSpec(
        num_layers=config.text_config.num_hidden_layers,
        batch=1,
        num_key_value_heads=config.text_config.num_key_value_heads,
        max_cache_len=max_cache_len,
        head_dim=config.text_config.head_dim,
        dtype="float32",
    )
    inputs = {
        "input_ids": dml.TensorSpec([1, 1], "int64"),
        "cos": dml.TensorSpec([1, 1, config.text_config.head_dim], "float32"),
        "sin": dml.TensorSpec([1, 1, config.text_config.head_dim], "float32"),
        "attention_mask": dml.TensorSpec([config.text_config.num_attention_heads, 1, max_cache_len + 1], "float32"),
        **static_kv_cache_input_specs(cache_spec),
    }

    spec = dml.trace(
        Qwen2_5_VLForConditionalGenerationDecodeStaticCache(config, weights),
        inputs=inputs,
        name="qwen2_5_vl_tiny_decode_static_cache",
    )
    validate_ir(shape_type_infer(deepcopy(spec.ir)))
    output_shapes = {output["name"]: output["shape"] for output in spec.ir["outputs"]}
    counts = Counter(node["op"] for node in spec.ir["nodes"])

    assert output_shapes["logits"] == [1, 1, config.text_config.vocab_size]
    assert output_shapes["new_key_0"] == [1, config.text_config.num_key_value_heads, 1, config.text_config.head_dim]
    assert output_shapes["new_value_0"] == [1, config.text_config.num_key_value_heads, 1, config.text_config.head_dim]
    assert _cache_concatenate_count(spec) == 2
    assert counts["bmm_rcr"] == 1
    assert counts["bmm_rrr"] == 1


def test_qwen2_5_vl_tiny_static_cache_decode_uses_flash_attention_bias_cache_path():
    base_config = _tiny_config()
    config = replace(
        base_config,
        text_config=replace(base_config.text_config, dtype="bfloat16", use_flash_attention=True, use_flash_attention_bias=True),
    )
    weights = _tiny_weights(config)
    max_cache_len = 5
    cache_spec = StaticKvCacheSpec(
        num_layers=config.text_config.num_hidden_layers,
        batch=1,
        num_key_value_heads=config.text_config.num_key_value_heads,
        max_cache_len=max_cache_len,
        head_dim=config.text_config.head_dim,
        dtype="bfloat16",
    )
    inputs = {
        "input_ids": dml.TensorSpec([1, 1], "int64"),
        "cos": dml.TensorSpec([1, 1, config.text_config.head_dim], "bfloat16"),
        "sin": dml.TensorSpec([1, 1, config.text_config.head_dim], "bfloat16"),
        "attention_mask": dml.TensorSpec([config.text_config.num_attention_heads, 1, max_cache_len], "bfloat16"),
        "cache_seqlens": dml.TensorSpec([1], "int32"),
        **static_kv_cache_input_specs(cache_spec),
    }

    spec = dml.trace(
        Qwen2_5_VLForConditionalGenerationDecodeStaticCache(config, weights),
        inputs=inputs,
        name="qwen2_5_vl_tiny_decode_static_cache_flash",
    )
    validate_ir(shape_type_infer(deepcopy(spec.ir)))
    counts = Counter(node["op"] for node in spec.ir["nodes"])

    assert counts["flash_attention_static_kv_cache_bias"] == 1
    assert counts["flash_attention_static_kv_cache"] == 0
    assert _cache_concatenate_count(spec) == 0
    assert counts["bmm_rcr"] == 0
    assert counts["bmm_rrr"] == 0


def test_qwen2_5_vl_stitch_op_reference_handles_discontiguous_image_spans():
    class StitchModule(dml.nn.Module):
        def forward(self, input_ids, inputs_embeds, image_features):
            stitched = dml.ops.qwen2_5_vl_stitch_image_features(
                input_ids,
                inputs_embeds,
                image_features,
                image_token_id=30,
            )
            return {"stitched": dml.ops.output(stitched, "stitched")}

    spec = dml.trace(
        StitchModule(),
        inputs={
            "input_ids": dml.TensorSpec([1, 5], "int64"),
            "inputs_embeds": dml.TensorSpec([1, 5, 2], "float32"),
            "image_features": dml.TensorSpec([3, 2], "float32"),
        },
        name="qwen2_5_vl_stitch_reference",
    )

    result = reference_numpy(
        spec,
        {
            "input_ids": np.asarray([[30, 1, 30, 2, 30]], dtype=np.int64),
            "inputs_embeds": np.zeros((1, 5, 2), dtype=np.float32),
            "image_features": np.asarray([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]], dtype=np.float32),
        },
    )

    np.testing.assert_array_equal(
        result["stitched"],
        np.asarray([[[1.0, 1.5], [0.0, 0.0], [2.0, 2.5], [0.0, 0.0], [3.0, 3.5]]], dtype=np.float32),
    )


def test_qwen2_5_vl_stitch_op_rocm_contract_accepts_int32_input_ids():
    class StitchModule(dml.nn.Module):
        def forward(self, input_ids, inputs_embeds, image_features):
            stitched = dml.ops.qwen2_5_vl_stitch_image_features(
                input_ids,
                inputs_embeds,
                image_features,
                image_token_id=30,
            )
            return {"stitched": dml.ops.output(stitched, "stitched")}

    spec = dml.trace(
        StitchModule(),
        inputs={
            "input_ids": dml.TensorSpec([2, 5], "int32"),
            "inputs_embeds": dml.TensorSpec([2, 5, 4], "bfloat16"),
            "image_features": dml.TensorSpec([4, 4], "bfloat16"),
        },
        name="qwen2_5_vl_stitch_rocm_contract",
    )

    _validate_mvp_runtime_contract(spec.ir, dml.Target("rocm"))


def _tiny_config() -> Qwen2_5_VLConfig:
    text = Qwen2_5_VLTextConfig(
        vocab_size=32,
        hidden_size=12,
        intermediate_size=10,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        rope_parameters={"rope_type": "default", "mrope_section": [1, 1, 1], "rope_theta": 10000.0},
        dtype="float32",
        use_flash_attention=False,
    )
    vision = Qwen2_5_VLVisionConfig(
        depth=1,
        hidden_size=8,
        intermediate_size=12,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=12,
        window_size=4,
        fullatt_block_indexes=(0,),
        dtype="float32",
        use_flash_attention=False,
    )
    return Qwen2_5_VLConfig(text_config=text, vision_config=vision, image_token_id=30, video_token_id=31)


def _cache_concatenate_count(spec) -> int:
    return sum(1 for node in spec.ir["nodes"] if node["op"] == "concatenate" and node.get("attrs", {}).get("dim") == 2)


def _prefill_fixture(config: Qwen2_5_VLConfig) -> dict[str, np.ndarray]:
    image_grid_thw = np.asarray([[1, 2, 2]], dtype=np.int64)
    input_ids = np.asarray([[1, config.image_token_id, 2]], dtype=np.int64)
    mm_token_type_ids = np.asarray([[0, 1, 0]], dtype=np.int64)
    text_position_ids, _ = qwen2_5_vl_rope_index(
        input_ids,
        mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=config.vision_config.spatial_merge_size,
    )
    text_cos, text_sin = qwen2_5_vl_text_rope_embeddings(text_position_ids, config.text_config)
    vision_position_ids = qwen2_5_vl_vision_position_ids(image_grid_thw, config.vision_config.spatial_merge_size)
    vision_cos, vision_sin = qwen2_5_vl_vision_rope_embeddings(vision_position_ids, head_dim=config.vision_config.head_dim)
    pixel_values = np.arange(4 * config.vision_config.patch_dim, dtype=np.float32).reshape(4, -1) / 100.0
    seq_len = input_ids.shape[1]
    attention_mask = np.triu(
        np.full((config.text_config.num_attention_heads, seq_len, seq_len), -1.0e4, dtype=np.float32),
        k=1,
    )
    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "vision_cos": vision_cos,
        "vision_sin": vision_sin,
        "text_cos": text_cos,
        "text_sin": text_sin,
        "attention_mask": attention_mask,
    }


def _tiny_weights(config: Qwen2_5_VLConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260607)
    weights: dict[str, np.ndarray] = {}

    def add(name: str, shape: tuple[int, ...]) -> None:
        weights[name] = rng.normal(0.0, 0.02, shape).astype(np.float32)

    text = config.text_config
    add("model.embed_tokens.weight", (text.vocab_size, text.hidden_size))
    add("model.norm.weight", (text.hidden_size,))
    for idx in range(text.num_hidden_layers):
        prefix = f"model.layers.{idx}"
        add(f"{prefix}.input_layernorm.weight", (text.hidden_size,))
        add(f"{prefix}.self_attn.q_proj.weight", (text.q_proj_size, text.hidden_size))
        add(f"{prefix}.self_attn.q_proj.bias", (text.q_proj_size,))
        add(f"{prefix}.self_attn.k_proj.weight", (text.kv_proj_size, text.hidden_size))
        add(f"{prefix}.self_attn.k_proj.bias", (text.kv_proj_size,))
        add(f"{prefix}.self_attn.v_proj.weight", (text.kv_proj_size, text.hidden_size))
        add(f"{prefix}.self_attn.v_proj.bias", (text.kv_proj_size,))
        add(f"{prefix}.self_attn.o_proj.weight", (text.hidden_size, text.q_proj_size))
        add(f"{prefix}.post_attention_layernorm.weight", (text.hidden_size,))
        add(f"{prefix}.mlp.gate_proj.weight", (text.intermediate_size, text.hidden_size))
        add(f"{prefix}.mlp.up_proj.weight", (text.intermediate_size, text.hidden_size))
        add(f"{prefix}.mlp.down_proj.weight", (text.hidden_size, text.intermediate_size))

    vision = config.vision_config
    add("visual.patch_embed.proj.weight", (vision.hidden_size, vision.in_channels, vision.temporal_patch_size, vision.patch_size, vision.patch_size))
    add("visual.merger.ln_q.weight", (vision.hidden_size,))
    add("visual.merger.mlp.0.weight", (vision.hidden_size * vision.spatial_merge_unit, vision.hidden_size * vision.spatial_merge_unit))
    add("visual.merger.mlp.0.bias", (vision.hidden_size * vision.spatial_merge_unit,))
    add("visual.merger.mlp.2.weight", (vision.out_hidden_size, vision.hidden_size * vision.spatial_merge_unit))
    add("visual.merger.mlp.2.bias", (vision.out_hidden_size,))
    for idx in range(vision.depth):
        prefix = f"visual.blocks.{idx}"
        add(f"{prefix}.norm1.weight", (vision.hidden_size,))
        add(f"{prefix}.attn.qkv.weight", (vision.hidden_size * 3, vision.hidden_size))
        add(f"{prefix}.attn.qkv.bias", (vision.hidden_size * 3,))
        add(f"{prefix}.attn.proj.weight", (vision.hidden_size, vision.hidden_size))
        add(f"{prefix}.attn.proj.bias", (vision.hidden_size,))
        add(f"{prefix}.norm2.weight", (vision.hidden_size,))
        add(f"{prefix}.mlp.gate_proj.weight", (vision.intermediate_size, vision.hidden_size))
        add(f"{prefix}.mlp.gate_proj.bias", (vision.intermediate_size,))
        add(f"{prefix}.mlp.up_proj.weight", (vision.intermediate_size, vision.hidden_size))
        add(f"{prefix}.mlp.up_proj.bias", (vision.intermediate_size,))
        add(f"{prefix}.mlp.down_proj.weight", (vision.hidden_size, vision.intermediate_size))
        add(f"{prefix}.mlp.down_proj.bias", (vision.hidden_size,))
    return weights
