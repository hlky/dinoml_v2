from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from tools import glm_ocr_benchmark_common as glm_ocr_common
from tools import benchmark_glm_ocr_dinoml_real_image_cached_generate as glm_ocr_tool
from tools import benchmark_glm_ocr_transformers_real_image as transformers_tool


def _config(*, use_flash_attention_bias: bool = True, use_flash_attention: bool = True):
    return SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=2,
            num_key_value_heads=1,
            num_attention_heads=2,
            head_dim=4,
            vocab_size=32,
            dtype="bfloat16",
            use_flash_attention=use_flash_attention,
            use_flash_attention_bias=use_flash_attention_bias,
        ),
        vision_config=SimpleNamespace(dtype="bfloat16"),
    )


def _write_artifact(tmp_path: Path, metadata: dict, *, ops: set[str], graph_ops: list[str] | None = None) -> Path:
    artifact = tmp_path / "artifact.dinoml"
    artifact.mkdir()
    (artifact / "module.so").write_bytes(b"")
    (artifact / "manifest.json").write_text(json.dumps({"files": {"module": "module.so"}}), encoding="utf-8")
    (artifact / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (artifact / "kernel_manifest.json").write_text(
        json.dumps({"required_kernels": [{"op": op} for op in sorted(ops)]}),
        encoding="utf-8",
    )
    (artifact / "graph.dinoir.json").write_text(
        json.dumps({"nodes": [{"op": op} for op in (graph_ops or [])]}),
        encoding="utf-8",
    )
    return artifact


def _items(names: list[str], *, shape: list[int] | None = None, dtype: str = "bfloat16") -> list[dict[str, object]]:
    return [
        {"name": name, "shape": [1] if shape is None else list(shape), "dtype": dtype}
        for name in names
    ]


def test_real_image_benchmark_defaults_use_same_image():
    assert glm_ocr_tool.DEFAULT_IMAGE == glm_ocr_common.DEFAULT_IMAGE
    assert transformers_tool.DEFAULT_IMAGE == glm_ocr_common.DEFAULT_IMAGE
    assert glm_ocr_tool.DEFAULT_IMAGE == transformers_tool.DEFAULT_IMAGE


def test_real_image_benchmark_longest_side_resize_preserves_aspect_ratio():
    image = Image.new("RGB", (2496, 3150))

    resized = glm_ocr_common.resize_image_longest_side(image, 1024)

    assert resized.size == (811, 1024)


def test_real_image_benchmark_longest_side_resize_rejects_non_positive_value():
    image = Image.new("RGB", (10, 20))

    with pytest.raises(ValueError, match="--longest-side must be positive"):
        glm_ocr_common.resize_image_longest_side(image, 0)


def test_prefill_artifact_rejects_stale_mask_input_artifact(tmp_path: Path):
    config = _config(use_flash_attention_bias=True)
    outputs = ["logits"]
    for layer_idx in range(config.text_config.num_hidden_layers):
        outputs.extend([f"present_key_{layer_idx}", f"present_value_{layer_idx}"])
    artifact = _write_artifact(
        tmp_path,
        {
            "inputs": _items(["input_ids", "pixel_values", "vision_cos", "vision_sin", "text_cos", "text_sin", "attention_mask"]),
            "outputs": _items(outputs),
        },
        ops={"flash_attention"},
    )

    assert not glm_ocr_tool._prefill_artifact_compatible(artifact, config)


def test_prefill_artifact_rejects_stale_rope_input_artifact(tmp_path: Path):
    config = _config(use_flash_attention_bias=True)
    outputs = ["logits"]
    for layer_idx in range(config.text_config.num_hidden_layers):
        outputs.extend([f"present_key_{layer_idx}", f"present_value_{layer_idx}"])
    artifact = _write_artifact(
        tmp_path,
        {
            "inputs": _items(["input_ids", "pixel_values", "vision_cos", "vision_sin", "text_cos", "text_sin"]),
            "outputs": _items(outputs),
        },
        ops={"flash_attention"},
        graph_ops=["swiglu", "swiglu"],
    )

    assert not glm_ocr_tool._prefill_artifact_compatible(artifact, config)


def test_prefill_artifact_rejects_stale_vision_swiglu_graph(tmp_path: Path):
    config = _config(use_flash_attention_bias=True)
    outputs = ["logits"]
    for layer_idx in range(config.text_config.num_hidden_layers):
        outputs.extend([f"present_key_{layer_idx}", f"present_value_{layer_idx}"])
    metadata = {
        "inputs": _items(["input_ids", "pixel_values"]),
        "outputs": _items(outputs),
    }
    artifact = _write_artifact(
        tmp_path,
        metadata,
        ops={"flash_attention"},
        graph_ops=["swiglu", "swiglu", "swiglu"],
    )

    assert not glm_ocr_tool._prefill_artifact_compatible(artifact, config)


def test_prefill_artifact_accepts_current_text_swiglu_graph(tmp_path: Path):
    config = _config(use_flash_attention_bias=True)
    outputs = ["logits"]
    for layer_idx in range(config.text_config.num_hidden_layers):
        outputs.extend([f"present_key_{layer_idx}", f"present_value_{layer_idx}"])
    metadata = {
        "inputs": _items(["input_ids", "pixel_values"]),
        "outputs": _items(outputs),
    }
    artifact = _write_artifact(
        tmp_path,
        metadata,
        ops={"flash_attention"},
        graph_ops=["swiglu", "swiglu"],
    )

    assert glm_ocr_tool._prefill_artifact_compatible(artifact, config)


def test_prefill_artifact_accepts_masked_ck_bias_graph(tmp_path: Path):
    config = _config(use_flash_attention_bias=True)
    prompt_len = 5
    outputs = ["logits"]
    for layer_idx in range(config.text_config.num_hidden_layers):
        outputs.extend([f"present_key_{layer_idx}", f"present_value_{layer_idx}"])
    metadata = {
        "inputs": [
            {"name": "input_ids", "shape": [1, prompt_len], "dtype": "int64"},
            {"name": "pixel_values", "shape": [8, 1176], "dtype": "bfloat16"},
            {"name": "attention_mask", "shape": [2, prompt_len, prompt_len], "dtype": "bfloat16"},
        ],
        "outputs": [
            {"name": "logits", "shape": [1, 1, 32], "dtype": "bfloat16"},
            {"name": "present_key_0", "shape": [1, 1, prompt_len, 4], "dtype": "bfloat16"},
            {"name": "present_value_0", "shape": [1, 1, prompt_len, 4], "dtype": "bfloat16"},
            {"name": "present_key_1", "shape": [1, 1, prompt_len, 4], "dtype": "bfloat16"},
            {"name": "present_value_1", "shape": [1, 1, prompt_len, 4], "dtype": "bfloat16"},
        ],
    }
    artifact = _write_artifact(tmp_path, metadata, ops={"flash_attention_bias"}, graph_ops=["swiglu", "swiglu"])
    expected_inputs = {
        "input_ids": np.zeros((1, prompt_len), dtype=np.int64),
        "pixel_values": np.zeros((8, 1176), dtype=np.uint16),
        "vision_cos": np.zeros((8, 64), dtype=np.float32),
        "vision_sin": np.zeros((8, 64), dtype=np.float32),
        "text_cos": np.zeros((1, prompt_len, 4), dtype=np.uint16),
        "text_sin": np.zeros((1, prompt_len, 4), dtype=np.uint16),
        "attention_mask": np.zeros((2, prompt_len, prompt_len), dtype=np.uint16),
    }

    assert glm_ocr_tool._prefill_artifact_compatible(artifact, config, expected_inputs=expected_inputs)


def test_prefill_artifact_rejects_masked_graph_without_ck_bias(tmp_path: Path):
    config = _config(use_flash_attention_bias=True)
    prompt_len = 5
    outputs = ["logits"]
    for layer_idx in range(config.text_config.num_hidden_layers):
        outputs.extend([f"present_key_{layer_idx}", f"present_value_{layer_idx}"])
    metadata = {
        "inputs": [
            {"name": "input_ids", "shape": [1, prompt_len], "dtype": "int64"},
            {"name": "pixel_values", "shape": [8, 1176], "dtype": "bfloat16"},
            {"name": "attention_mask", "shape": [2, prompt_len, prompt_len], "dtype": "bfloat16"},
        ],
        "outputs": _items(outputs),
    }
    artifact = _write_artifact(tmp_path, metadata, ops={"flash_attention"}, graph_ops=["swiglu", "swiglu"])
    expected_inputs = {
        "input_ids": np.zeros((1, prompt_len), dtype=np.int64),
        "pixel_values": np.zeros((8, 1176), dtype=np.uint16),
        "vision_cos": np.zeros((8, 64), dtype=np.float32),
        "vision_sin": np.zeros((8, 64), dtype=np.float32),
        "text_cos": np.zeros((1, prompt_len, 4), dtype=np.uint16),
        "text_sin": np.zeros((1, prompt_len, 4), dtype=np.uint16),
        "attention_mask": np.zeros((2, prompt_len, prompt_len), dtype=np.uint16),
    }

    assert not glm_ocr_tool._prefill_artifact_compatible(artifact, config, expected_inputs=expected_inputs)


def test_prefill_artifact_rejects_stale_shape_when_expected_inputs_are_available(tmp_path: Path):
    config = _config(use_flash_attention_bias=True)
    outputs = ["logits"]
    for layer_idx in range(config.text_config.num_hidden_layers):
        outputs.extend([f"present_key_{layer_idx}", f"present_value_{layer_idx}"])
    artifact = _write_artifact(
        tmp_path,
        {
            "inputs": [
                {"name": "input_ids", "shape": [1, 3], "dtype": "int64"},
                {"name": "pixel_values", "shape": [4, 1176], "dtype": "bfloat16"},
            ],
            "outputs": [
                {"name": "logits", "shape": [1, 1, 32], "dtype": "bfloat16"},
                {"name": "present_key_0", "shape": [1, 1, 3, 4], "dtype": "bfloat16"},
                {"name": "present_value_0", "shape": [1, 1, 3, 4], "dtype": "bfloat16"},
                {"name": "present_key_1", "shape": [1, 1, 3, 4], "dtype": "bfloat16"},
                {"name": "present_value_1", "shape": [1, 1, 3, 4], "dtype": "bfloat16"},
            ],
        },
        ops={"flash_attention"},
        graph_ops=["swiglu", "swiglu"],
    )
    expected_inputs = {
        "input_ids": np.zeros((1, 5), dtype=np.int64),
        "pixel_values": np.zeros((8, 1176), dtype=np.uint16),
        "vision_cos": np.zeros((8, 64), dtype=np.float32),
        "vision_sin": np.zeros((8, 64), dtype=np.float32),
        "text_cos": np.zeros((1, 5, 4), dtype=np.uint16),
        "text_sin": np.zeros((1, 5, 4), dtype=np.uint16),
    }

    assert not glm_ocr_tool._prefill_artifact_compatible(artifact, config, expected_inputs=expected_inputs)


def test_session_decode_artifact_rejects_external_cache_artifact(tmp_path: Path):
    config = _config()
    metadata = {
        "inputs": _items(
            [
                "input_ids",
                "cos",
                "sin",
                "attention_mask",
                "cache_seqlens",
                "past_key_0",
                "past_value_0",
                "past_key_1",
                "past_value_1",
            ]
        ),
        "outputs": _items(["logits", "new_key_0", "new_value_0", "new_key_1", "new_value_1"]),
        "states": [],
    }
    artifact = _write_artifact(tmp_path, metadata, ops={"flash_attention_static_kv_cache_bias"})

    assert not glm_ocr_tool._decode_artifact_compatible(
        artifact,
        config,
        8,
        use_flash_static_kv_cache=True,
        use_session_static_kv_cache=True,
    )


def test_session_decode_artifact_accepts_state_cache_artifact(tmp_path: Path):
    config = _config()
    state_names = ["past_key_0", "past_value_0", "past_key_1", "past_value_1"]
    metadata = {
        "inputs": [
            {"name": "input_ids", "shape": [1, 1], "dtype": "int64"},
            {"name": "cos", "shape": [1, 1, 4], "dtype": "bfloat16"},
            {"name": "sin", "shape": [1, 1, 4], "dtype": "bfloat16"},
        ],
        "outputs": _items(["logits"], shape=[1, 1, 32]),
        "states": [
            *_items(state_names, shape=[1, 1, 8, 4]),
            {"name": "cache_seqlens", "shape": [1], "dtype": "int32"},
        ],
    }
    artifact = _write_artifact(tmp_path, metadata, ops={"flash_attention_static_kv_cache"})

    assert glm_ocr_tool._decode_artifact_compatible(
        artifact,
        config,
        8,
        use_flash_static_kv_cache=True,
        use_session_static_kv_cache=True,
    )


def test_session_decode_artifact_accepts_state_cache_masked_bias_artifact(tmp_path: Path):
    config = _config()
    state_names = ["past_key_0", "past_value_0", "past_key_1", "past_value_1"]
    metadata = {
        "inputs": [
            {"name": "input_ids", "shape": [1, 1], "dtype": "int64"},
            {"name": "cos", "shape": [1, 1, 4], "dtype": "bfloat16"},
            {"name": "sin", "shape": [1, 1, 4], "dtype": "bfloat16"},
            {"name": "attention_mask", "shape": [2, 1, 8], "dtype": "bfloat16"},
        ],
        "outputs": _items(["logits"], shape=[1, 1, 32]),
        "states": [
            *_items(state_names, shape=[1, 1, 8, 4]),
            {"name": "cache_seqlens", "shape": [1], "dtype": "int32"},
        ],
    }
    artifact = _write_artifact(tmp_path, metadata, ops={"flash_attention_static_kv_cache_bias"})

    assert glm_ocr_tool._decode_artifact_compatible(
        artifact,
        config,
        8,
        use_flash_static_kv_cache=True,
        use_session_static_kv_cache=True,
        use_decode_attention_mask=True,
    )


def test_session_decode_artifact_rejects_masked_state_cache_without_ck_bias(tmp_path: Path):
    config = _config()
    state_names = ["past_key_0", "past_value_0", "past_key_1", "past_value_1"]
    metadata = {
        "inputs": [
            {"name": "input_ids", "shape": [1, 1], "dtype": "int64"},
            {"name": "cos", "shape": [1, 1, 4], "dtype": "bfloat16"},
            {"name": "sin", "shape": [1, 1, 4], "dtype": "bfloat16"},
            {"name": "attention_mask", "shape": [2, 1, 8], "dtype": "bfloat16"},
        ],
        "outputs": _items(["logits"], shape=[1, 1, 32]),
        "states": [
            *_items(state_names, shape=[1, 1, 8, 4]),
            {"name": "cache_seqlens", "shape": [1], "dtype": "int32"},
        ],
    }
    artifact = _write_artifact(tmp_path, metadata, ops={"flash_attention_static_kv_cache"})

    assert not glm_ocr_tool._decode_artifact_compatible(
        artifact,
        config,
        8,
        use_flash_static_kv_cache=True,
        use_session_static_kv_cache=True,
        use_decode_attention_mask=True,
    )


def test_session_decode_artifact_rejects_stale_attention_mask_input(tmp_path: Path):
    config = _config()
    state_names = ["past_key_0", "past_value_0", "past_key_1", "past_value_1"]
    metadata = {
        "inputs": [
            {"name": "input_ids", "shape": [1, 1], "dtype": "int64"},
            {"name": "cos", "shape": [1, 1, 4], "dtype": "bfloat16"},
            {"name": "sin", "shape": [1, 1, 4], "dtype": "bfloat16"},
            {"name": "attention_mask", "shape": [2, 1, 7], "dtype": "bfloat16"},
        ],
        "outputs": _items(["logits"]),
        "states": [
            *_items(state_names, shape=[1, 1, 8, 4]),
            {"name": "cache_seqlens", "shape": [1], "dtype": "int32"},
        ],
    }
    artifact = _write_artifact(tmp_path, metadata, ops={"flash_attention_static_kv_cache"})

    assert not glm_ocr_tool._decode_artifact_compatible(
        artifact,
        config,
        8,
        use_flash_static_kv_cache=True,
        use_session_static_kv_cache=True,
    )


def test_session_static_kv_cache_selection_does_not_require_bias_flag():
    args = SimpleNamespace(target="rocm")

    assert glm_ocr_tool._use_flash_static_kv_cache(args, _config(use_flash_attention_bias=False))
    assert glm_ocr_tool._use_session_static_kv_cache(args, _config(use_flash_attention_bias=False))
    assert not glm_ocr_tool._use_flash_static_kv_cache(args, _config(use_flash_attention=False))
    assert not glm_ocr_tool._use_session_static_kv_cache(args, _config(use_flash_attention=False))


def test_decode_run_inputs_can_include_flash_static_attention_mask():
    config = _config(use_flash_attention_bias=True)
    prefill_inputs = {
        "text_cos": np.zeros((1, 8, 4), dtype=np.uint16),
        "text_sin": np.zeros((1, 8, 4), dtype=np.uint16),
    }

    inputs = glm_ocr_tool._decode_run_inputs(
        prefill_inputs,
        config,
        8,
        next_id=7,
        position=3,
        cache={},
        use_flash_static_kv_cache=True,
        use_session_static_kv_cache=True,
        use_decode_attention_mask=True,
    )

    assert set(inputs) == {"input_ids", "cos", "sin", "attention_mask"}
    assert inputs["attention_mask"].shape == (2, 1, 8)
    assert inputs["attention_mask"].dtype == np.uint16
