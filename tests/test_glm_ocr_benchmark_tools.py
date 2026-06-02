from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from tools import benchmark_glm_ocr_dinoml_real_image_cached_generate as glm_ocr_tool


def _config(*, use_flash_attention_bias: bool = True):
    return SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=2,
            num_key_value_heads=1,
            head_dim=4,
            dtype="bfloat16",
            use_flash_attention_bias=use_flash_attention_bias,
        )
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


def test_prefill_artifact_requires_flash_attention_bias_when_config_uses_bias(tmp_path: Path):
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


def test_prefill_artifact_rejects_stale_vision_swiglu_graph(tmp_path: Path):
    config = _config(use_flash_attention_bias=True)
    outputs = ["logits"]
    for layer_idx in range(config.text_config.num_hidden_layers):
        outputs.extend([f"present_key_{layer_idx}", f"present_value_{layer_idx}"])
    metadata = {
        "inputs": _items(["input_ids", "pixel_values", "vision_cos", "vision_sin", "text_cos", "text_sin", "attention_mask"]),
        "outputs": _items(outputs),
    }
    artifact = _write_artifact(
        tmp_path,
        metadata,
        ops={"flash_attention_bias"},
        graph_ops=["swiglu", "swiglu", "swiglu"],
    )

    assert not glm_ocr_tool._prefill_artifact_compatible(artifact, config)


def test_prefill_artifact_accepts_current_text_swiglu_graph(tmp_path: Path):
    config = _config(use_flash_attention_bias=True)
    outputs = ["logits"]
    for layer_idx in range(config.text_config.num_hidden_layers):
        outputs.extend([f"present_key_{layer_idx}", f"present_value_{layer_idx}"])
    metadata = {
        "inputs": _items(["input_ids", "pixel_values", "vision_cos", "vision_sin", "text_cos", "text_sin", "attention_mask"]),
        "outputs": _items(outputs),
    }
    artifact = _write_artifact(
        tmp_path,
        metadata,
        ops={"flash_attention_bias"},
        graph_ops=["swiglu", "swiglu"],
    )

    assert glm_ocr_tool._prefill_artifact_compatible(artifact, config)


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
        "inputs": _items(["input_ids", "cos", "sin", "attention_mask", "cache_seqlens"]),
        "outputs": _items(["logits"]),
        "states": _items(state_names, shape=[1, 1, 8, 4]),
    }
    artifact = _write_artifact(tmp_path, metadata, ops={"flash_attention_static_kv_cache_bias"})

    assert glm_ocr_tool._decode_artifact_compatible(
        artifact,
        config,
        8,
        use_flash_static_kv_cache=True,
        use_session_static_kv_cache=True,
    )
