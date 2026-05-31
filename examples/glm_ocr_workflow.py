from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

import dinoml as dml
from dinoml.ir import ModelSpec
from dinoml.models.glm_ocr import (
    GlmOcrConfig,
    GlmOcrForConditionalGeneration,
    GlmOcrForConditionalGenerationDecode,
    GlmOcrTextConfig,
    GlmOcrVisionConfig,
    GlmOcrVisionModel,
)
from dinoml.reference import reference_numpy


BATCH = 1
SEQ_LEN = 3
VISION_PATCHES = 4
PAST_LEN = 2


def build_config() -> GlmOcrConfig:
    text = GlmOcrTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=10,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=6,
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [1, 1, 1],
            "partial_rotary_factor": 1.0,
            "rope_theta": 10000.0,
        },
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


def build_weights() -> dict[str, np.ndarray]:
    config = build_config()
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
    add(
        "model.visual.patch_embed.proj.weight",
        (vision.hidden_size, vision.in_channels, vision.temporal_patch_size, vision.patch_size, vision.patch_size),
    )
    add("model.visual.patch_embed.proj.bias", (vision.hidden_size,))
    add("model.visual.post_layernorm.weight", (vision.hidden_size,))
    add(
        "model.visual.downsample.weight",
        (vision.out_hidden_size, vision.hidden_size, vision.spatial_merge_size, vision.spatial_merge_size),
    )
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


WEIGHTS = build_weights()


def build_spec() -> ModelSpec:
    config = build_config()
    return dml.trace(
        GlmOcrForConditionalGeneration(config, WEIGHTS),
        inputs={
            "input_ids": dml.TensorSpec([BATCH, SEQ_LEN], "int64"),
            "cos": dml.TensorSpec([BATCH, SEQ_LEN, config.text_config.head_dim], "float32"),
            "sin": dml.TensorSpec([BATCH, SEQ_LEN, config.text_config.head_dim], "float32"),
            "attention_mask": dml.TensorSpec([config.text_config.num_attention_heads, SEQ_LEN, SEQ_LEN], "float32"),
        },
        name="glm_ocr_text_prefill_smoke",
    )


def build_vision_spec() -> ModelSpec:
    config = build_config()
    return dml.trace(
        GlmOcrVisionModel(config.vision_config, WEIGHTS),
        inputs={
            "pixel_values": dml.TensorSpec([VISION_PATCHES, config.vision_config.patch_dim], "float32"),
            "cos": dml.TensorSpec([VISION_PATCHES, config.vision_config.head_dim], "float32"),
            "sin": dml.TensorSpec([VISION_PATCHES, config.vision_config.head_dim], "float32"),
        },
        name="glm_ocr_vision_smoke",
    )


def build_decode_spec() -> ModelSpec:
    config = build_config()
    inputs = {
        "input_ids": dml.TensorSpec([BATCH, 1], "int64"),
        "cos": dml.TensorSpec([BATCH, 1, config.text_config.head_dim], "float32"),
        "sin": dml.TensorSpec([BATCH, 1, config.text_config.head_dim], "float32"),
        "attention_mask": dml.TensorSpec(
            [config.text_config.num_attention_heads, 1, PAST_LEN + 1],
            "float32",
        ),
    }
    for layer_idx in range(config.text_config.num_hidden_layers):
        inputs[f"past_key_{layer_idx}"] = dml.TensorSpec(
            [BATCH, config.text_config.num_key_value_heads, PAST_LEN, config.text_config.head_dim],
            "float32",
        )
        inputs[f"past_value_{layer_idx}"] = dml.TensorSpec(
            [BATCH, config.text_config.num_key_value_heads, PAST_LEN, config.text_config.head_dim],
            "float32",
        )
    return dml.trace(
        GlmOcrForConditionalGenerationDecode(config, WEIGHTS),
        inputs=inputs,
        name="glm_ocr_decode_smoke",
    )


def build_validation_inputs() -> dict[str, np.ndarray]:
    config = build_config()
    return {
        "input_ids": np.asarray([[1, config.image_token_id, 2]], dtype=np.int64),
        "cos": np.ones((BATCH, SEQ_LEN, config.text_config.head_dim), dtype=np.float32),
        "sin": np.zeros((BATCH, SEQ_LEN, config.text_config.head_dim), dtype=np.float32),
        "attention_mask": np.triu(
            np.full((config.text_config.num_attention_heads, SEQ_LEN, SEQ_LEN), -1.0e4, dtype=np.float32),
            k=1,
        ),
    }


def inspect_workflow() -> dict[str, object]:
    text_spec = build_spec()
    vision_spec = build_vision_spec()
    decode_spec = build_decode_spec()
    return {
        "text_name": text_spec.name,
        "text_outputs": text_spec.ir["outputs"],
        "text_node_op_counts": dict(sorted(Counter(node["op"] for node in text_spec.ir["nodes"]).items())),
        "decode_name": decode_spec.name,
        "decode_outputs": decode_spec.ir["outputs"],
        "decode_node_op_counts": dict(sorted(Counter(node["op"] for node in decode_spec.ir["nodes"]).items())),
        "vision_name": vision_spec.name,
        "vision_outputs": vision_spec.ir["outputs"],
        "vision_node_op_counts": dict(sorted(Counter(node["op"] for node in vision_spec.ir["nodes"]).items())),
    }


def run_example() -> dict[str, object]:
    spec = build_spec()
    outputs = reference_numpy(spec, build_validation_inputs())
    summary = inspect_workflow()
    summary["logits_preview"] = np.round(outputs["logits"][0, -1, :5], 6).tolist()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the bounded GLM-OCR workflow smoke graphs.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = run_example()
    print(json.dumps(payload, indent=2, sort_keys=True) if args.json else payload)


if __name__ == "__main__":
    main()
