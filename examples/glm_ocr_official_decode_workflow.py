from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import dinoml as dml
from dinoml.ir import array_to_storage
from dinoml.models.glm_ocr import (
    GlmOcrForConditionalGenerationDecode,
    glm_ocr_config_from_transformers_dict,
    glm_ocr_required_text_weight_names,
    glm_ocr_weights_from_safetensors_file,
)


SNAPSHOT = Path(
    os.environ.get(
        "GLM_OCR_SNAPSHOT",
        r"C:\Users\user\.cache\huggingface\hub\models--zai-org--GLM-OCR\snapshots\ca5d8b3e287e52589e37c28385d9655ee4372f9d",
    )
)
PAST_LEN = int(os.environ.get("GLM_OCR_DECODE_PAST_LEN", "16"))
BATCH = int(os.environ.get("GLM_OCR_DECODE_BATCH", "1"))
DTYPE = os.environ.get("GLM_OCR_DTYPE", "bfloat16")


def build_config():
    payload = json.loads((SNAPSHOT / "config.json").read_text(encoding="utf-8"))
    return glm_ocr_config_from_transformers_dict(payload, dtype=DTYPE)


def build_weights():
    config = build_config()
    return glm_ocr_weights_from_safetensors_file(
        SNAPSHOT / "model.safetensors",
        config,
        dtype=config.text_config.dtype,
        required_names=glm_ocr_required_text_weight_names(config),
    )


def build_spec():
    config = build_config()
    inputs = {
        "input_ids": dml.TensorSpec([BATCH, 1], "int64"),
        "cos": dml.TensorSpec([BATCH, 1, config.text_config.head_dim], config.text_config.dtype),
        "sin": dml.TensorSpec([BATCH, 1, config.text_config.head_dim], config.text_config.dtype),
        "attention_mask": dml.TensorSpec(
            [BATCH * config.text_config.num_attention_heads, 1, PAST_LEN + 1],
            config.text_config.dtype,
        ),
    }
    for layer_idx in range(config.text_config.num_hidden_layers):
        inputs[f"past_key_{layer_idx}"] = dml.TensorSpec(
            [BATCH, config.text_config.num_key_value_heads, PAST_LEN, config.text_config.head_dim],
            config.text_config.dtype,
        )
        inputs[f"past_value_{layer_idx}"] = dml.TensorSpec(
            [BATCH, config.text_config.num_key_value_heads, PAST_LEN, config.text_config.head_dim],
            config.text_config.dtype,
        )
    return dml.trace(
        GlmOcrForConditionalGenerationDecode(config, build_weights()),
        inputs=inputs,
        name=f"glm_ocr_official_decode_b{BATCH}_past{PAST_LEN}",
    )


def build_validation_inputs() -> dict[str, np.ndarray]:
    config = build_config()
    rng = np.random.default_rng(20260530)
    inputs = {
        "input_ids": np.full((BATCH, 1), 42, dtype=np.int64),
        "cos": _float_input(np.ones((BATCH, 1, config.text_config.head_dim), dtype=np.float32), config.text_config.dtype),
        "sin": _float_input(np.zeros((BATCH, 1, config.text_config.head_dim), dtype=np.float32), config.text_config.dtype),
        "attention_mask": _float_input(
            np.zeros((BATCH * config.text_config.num_attention_heads, 1, PAST_LEN + 1), dtype=np.float32),
            config.text_config.dtype,
        ),
    }
    for layer_idx in range(config.text_config.num_hidden_layers):
        inputs[f"past_key_{layer_idx}"] = _float_input(
            rng.normal(
                0.0,
                0.01,
                (BATCH, config.text_config.num_key_value_heads, PAST_LEN, config.text_config.head_dim),
            ).astype(np.float32),
            config.text_config.dtype,
        )
        inputs[f"past_value_{layer_idx}"] = _float_input(
            rng.normal(
                0.0,
                0.01,
                (BATCH, config.text_config.num_key_value_heads, PAST_LEN, config.text_config.head_dim),
            ).astype(np.float32),
            config.text_config.dtype,
        )
    return inputs


def _float_input(values: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bfloat16":
        return array_to_storage(values, "bfloat16")
    return values.astype(dtype, copy=False)


def _np_dtype(dtype: str) -> np.dtype:
    if dtype == "bfloat16":
        return np.uint16
    return np.dtype(dtype)
