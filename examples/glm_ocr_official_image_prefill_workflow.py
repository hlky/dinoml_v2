from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import dinoml as dml
from dinoml.ir import ModelSpec, array_to_storage
from dinoml.models.glm_ocr import (
    GlmOcrForConditionalGenerationImagePrefill,
    glm_ocr_config_from_transformers_dict,
    glm_ocr_rope_index,
    glm_ocr_text_rope_embeddings,
    glm_ocr_vision_position_ids,
    glm_ocr_vision_rope_embeddings,
    glm_ocr_weights_from_safetensors_file,
)


SNAPSHOT = Path(
    os.environ.get(
        "GLM_OCR_SNAPSHOT",
        r"C:\Users\user\.cache\huggingface\hub\models--zai-org--GLM-OCR\snapshots\ca5d8b3e287e52589e37c28385d9655ee4372f9d",
    )
)
GRID_THW = tuple(int(part) for part in os.environ.get("GLM_OCR_IMAGE_GRID_THW", "1,8,8").split(","))
TEXT_TAIL_LEN = int(os.environ.get("GLM_OCR_TEXT_TAIL_LEN", "1"))
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
    )


def build_spec() -> ModelSpec:
    config = build_config()
    image_feature_count = _image_feature_count(config)
    patch_count = int(np.prod(GRID_THW))
    seq_len = 1 + image_feature_count + TEXT_TAIL_LEN
    return dml.trace(
        GlmOcrForConditionalGenerationImagePrefill(
            config,
            build_weights(),
            image_token_start=1,
            logits_to_keep=1,
        ),
        inputs={
            "input_ids": dml.TensorSpec([1, seq_len], "int64"),
            "pixel_values": dml.TensorSpec([patch_count, config.vision_config.patch_dim], config.vision_config.dtype),
            "vision_cos": dml.TensorSpec([patch_count, config.vision_config.head_dim], config.vision_config.dtype),
            "vision_sin": dml.TensorSpec([patch_count, config.vision_config.head_dim], config.vision_config.dtype),
            "text_cos": dml.TensorSpec([1, seq_len, config.text_config.head_dim], config.text_config.dtype),
            "text_sin": dml.TensorSpec([1, seq_len, config.text_config.head_dim], config.text_config.dtype),
            "attention_mask": dml.TensorSpec(
                [config.text_config.num_attention_heads, seq_len, seq_len],
                config.text_config.dtype,
            ),
        },
        name=f"glm_ocr_official_image_prefill_grid{GRID_THW[1]}x{GRID_THW[2]}_s{seq_len}",
    )


def build_validation_inputs() -> dict[str, np.ndarray]:
    config = build_config()
    image_grid_thw = np.asarray([GRID_THW], dtype=np.int64)
    image_feature_count = _image_feature_count(config)
    patch_count = int(np.prod(GRID_THW))
    seq_len = 1 + image_feature_count + TEXT_TAIL_LEN
    input_ids = np.concatenate(
        [
            np.asarray([42], dtype=np.int64),
            np.full((image_feature_count,), config.image_token_id, dtype=np.int64),
            np.full((TEXT_TAIL_LEN,), 43, dtype=np.int64),
        ]
    ).reshape(1, seq_len)
    mm_token_type_ids = np.concatenate(
        [
            np.zeros((1,), dtype=np.int64),
            np.ones((image_feature_count,), dtype=np.int64),
            np.zeros((TEXT_TAIL_LEN,), dtype=np.int64),
        ]
    ).reshape(1, seq_len)
    text_position_ids, _ = glm_ocr_rope_index(
        input_ids,
        mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=config.vision_config.spatial_merge_size,
    )
    text_cos, text_sin = glm_ocr_text_rope_embeddings(text_position_ids, config.text_config, dtype=config.text_config.dtype)
    vision_position_ids = glm_ocr_vision_position_ids(image_grid_thw, config.vision_config.spatial_merge_size)
    vision_cos, vision_sin = glm_ocr_vision_rope_embeddings(
        vision_position_ids,
        head_dim=config.vision_config.head_dim,
        dtype=config.vision_config.dtype,
    )
    rng = np.random.default_rng(20260530)
    pixel_values = rng.normal(0.0, 0.2, (patch_count, config.vision_config.patch_dim)).astype(np.float32)
    attention_mask = np.triu(
        np.full((config.text_config.num_attention_heads, seq_len, seq_len), -1.0e4, dtype=np.float32),
        k=1,
    )
    return {
        "input_ids": input_ids,
        "pixel_values": _float_input(pixel_values, config.vision_config.dtype),
        "vision_cos": _float_input(vision_cos, config.vision_config.dtype),
        "vision_sin": _float_input(vision_sin, config.vision_config.dtype),
        "text_cos": _float_input(text_cos, config.text_config.dtype),
        "text_sin": _float_input(text_sin, config.text_config.dtype),
        "attention_mask": _float_input(attention_mask, config.text_config.dtype),
    }


def _image_feature_count(config) -> int:
    grid_t, grid_h, grid_w = GRID_THW
    merge = config.vision_config.spatial_merge_size
    if grid_h % merge or grid_w % merge:
        raise ValueError("GLM_OCR_IMAGE_GRID_THW height and width must be divisible by spatial_merge_size")
    return int(grid_t * grid_h * grid_w // (merge * merge))


def _float_input(values: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bfloat16":
        if values.dtype == np.uint16:
            return np.ascontiguousarray(values)
        return array_to_storage(values.astype(np.float32, copy=False), "bfloat16")
    return values.astype(dtype, copy=False)
