from __future__ import annotations

from types import SimpleNamespace
import json

import numpy as np
import pytest
from PIL import Image

from tools import benchmark_glm_ocr_static_cache_pipeline as pipeline_tool
from tools import glm_ocr_benchmark_common as glm_ocr_common


def _config(*, dtype: str = "float16", num_layers: int = 1):
    return SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=num_layers,
            num_key_value_heads=1,
            num_attention_heads=2,
            head_dim=4,
            dtype=dtype,
        ),
        vision_config=SimpleNamespace(dtype=dtype),
    )


def _write_artifact_metadata(path, *, inputs: list[str], outputs: list[str]) -> None:
    path.mkdir()
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "inputs": [{"name": name} for name in inputs],
                "outputs": [{"name": name} for name in outputs],
            }
        ),
        encoding="utf-8",
    )


def test_static_cache_pipeline_defaults_use_shared_common_paths():
    assert pipeline_tool.DEFAULT_IMAGE == glm_ocr_common.DEFAULT_IMAGE
    assert pipeline_tool.DEFAULT_PROMPT == glm_ocr_common.DEFAULT_PROMPT
    assert pipeline_tool.DEFAULT_SNAPSHOT == glm_ocr_common.DEFAULT_SNAPSHOT


def test_real_image_benchmark_longest_side_resize_preserves_aspect_ratio():
    image = Image.new("RGB", (2496, 3150))

    resized = glm_ocr_common.resize_image_longest_side(image, 1024)

    assert resized.size == (811, 1024)


def test_real_image_benchmark_longest_side_resize_rejects_non_positive_value():
    image = Image.new("RGB", (10, 20))

    with pytest.raises(ValueError, match="--longest-side must be positive"):
        glm_ocr_common.resize_image_longest_side(image, 0)


@pytest.mark.parametrize(
    ("decode_outputs", "expected_mode"),
    (
        (["logits", "new_key_0", "new_value_0"], "static"),
        (["logits", "present_key_0", "present_value_0"], "dynamic"),
    ),
)
def test_validate_artifacts_accepts_dynamic_workflow_style_inputs(tmp_path, decode_outputs, expected_mode):
    prefill = tmp_path / "prefill.dinoml"
    decode = tmp_path / "decode.dinoml"
    _write_artifact_metadata(
        prefill,
        inputs=["input_ids", "pixel_values", "vision_cos", "vision_sin", "text_cos", "text_sin", "attention_mask"],
        outputs=["logits", "present_key_0", "present_value_0"],
    )
    _write_artifact_metadata(
        decode,
        inputs=(
            ["input_ids", "cos", "sin", "attention_mask", "cache_seqlens", "past_key_0", "past_value_0"]
            if expected_mode == "static"
            else ["input_ids", "cos", "sin", "attention_mask", "past_key_0", "past_value_0"]
        ),
        outputs=decode_outputs,
    )

    result = pipeline_tool.validate_artifacts(
        prefill_artifact=prefill,
        decode_artifact=decode,
    )

    assert result["decode_mode"] == expected_mode
    assert result["use_decode_attention_mask"] is True


@pytest.mark.parametrize(
    ("decode_mode", "expected_keys", "expected_cache_shape"),
    (
        (
            "static",
            {"input_ids", "cos", "sin", "cache_seqlens", "past_key_0", "past_value_0", "attention_mask"},
            (1, 1, 4, 4),
        ),
        (
            "dynamic",
            {"input_ids", "cos", "sin", "past_key_0", "past_value_0", "attention_mask"},
            (1, 1, 8, 4),
        ),
    ),
)
def test_decode_step_inputs_slice_dynamic_cache_and_rope_inputs(decode_mode, expected_keys, expected_cache_shape):
    config = _config()
    full_inputs = {
        "text_cos": np.arange(24, dtype=np.float16).reshape(1, 6, 4),
        "text_sin": np.arange(24, 48, dtype=np.float16).reshape(1, 6, 4),
    }
    cache = {
        "past_key_0": np.zeros((1, 1, 8, 4), dtype=np.float16),
        "past_value_0": np.zeros((1, 1, 8, 4), dtype=np.float16),
    }

    inputs = pipeline_tool.decode_step_inputs(
        full_inputs=full_inputs,
        config=config,
        cache=cache,
        next_id=7,
        position=3,
        decode_mode=decode_mode,
        use_attention_mask=True,
    )

    assert set(inputs) == expected_keys
    assert inputs["cos"].shape == (1, 1, 4)
    assert inputs["sin"].shape == (1, 1, 4)
    assert np.array_equal(inputs["cos"][0, 0], full_inputs["text_cos"][0, 3])
    assert np.array_equal(inputs["sin"][0, 0], full_inputs["text_sin"][0, 3])
    assert inputs["past_key_0"].shape == expected_cache_shape
    assert inputs["past_value_0"].shape == expected_cache_shape
    assert inputs["attention_mask"].shape == (2, 1, 4)
