from __future__ import annotations

import numpy as np

from examples.glm_ocr_workflow import BATCH, PAST_LEN, build_config, build_decode_spec


def build_spec():
    return build_decode_spec()


def build_validation_inputs() -> dict[str, np.ndarray]:
    config = build_config()
    inputs = {
        "input_ids": np.asarray([[2]], dtype=np.int64),
        "cos": np.ones((BATCH, 1, config.text_config.head_dim), dtype=np.float32),
        "sin": np.zeros((BATCH, 1, config.text_config.head_dim), dtype=np.float32),
        "attention_mask": np.zeros(
            (config.text_config.num_attention_heads, 1, PAST_LEN + 1),
            dtype=np.float32,
        ),
    }
    for layer_idx in range(config.text_config.num_hidden_layers):
        inputs[f"past_key_{layer_idx}"] = np.zeros(
            (BATCH, config.text_config.num_key_value_heads, PAST_LEN, config.text_config.head_dim),
            dtype=np.float32,
        )
        inputs[f"past_value_{layer_idx}"] = np.zeros(
            (BATCH, config.text_config.num_key_value_heads, PAST_LEN, config.text_config.head_dim),
            dtype=np.float32,
        )
    return inputs
