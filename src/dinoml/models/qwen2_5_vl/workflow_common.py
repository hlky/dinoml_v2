from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from dinoml.ir import ModelSpec, array_to_storage

from .qwen2_5_vl import (
    qwen2_5_vl_config_from_transformers_dict,
    qwen2_5_vl_weights_from_safetensors_file,
    qwen2_5_vl_weights_from_safetensors_index,
)


def resolve_snapshot_paths(
    *,
    snapshot: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
) -> tuple[Path, Path, Path]:
    resolved_snapshot = Path(snapshot)
    resolved_config_path = Path(config_path) if config_path is not None else resolved_snapshot / "config.json"
    resolved_checkpoint_path = (
        Path(checkpoint_path) if checkpoint_path is not None else _default_checkpoint_path(resolved_snapshot)
    )
    return resolved_snapshot, resolved_config_path, resolved_checkpoint_path


def load_qwen2_5_vl_config(
    *,
    snapshot: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    dtype: str = "bfloat16",
):
    del checkpoint_path
    _, resolved_config_path, _ = resolve_snapshot_paths(snapshot=snapshot, config_path=config_path)
    payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    return qwen2_5_vl_config_from_transformers_dict(payload, dtype=str(dtype))


def load_qwen2_5_vl_weights(
    *,
    config,
    snapshot: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    required_names: Sequence[str] | None = None,
):
    del config_path
    _, _, resolved_checkpoint_path = resolve_snapshot_paths(snapshot=snapshot, checkpoint_path=checkpoint_path)
    if resolved_checkpoint_path.name.endswith(".index.json"):
        return qwen2_5_vl_weights_from_safetensors_index(
            resolved_checkpoint_path,
            config,
            dtype=config.text_config.dtype,
            required_names=required_names,
        )
    return qwen2_5_vl_weights_from_safetensors_file(
        resolved_checkpoint_path,
        config,
        dtype=config.text_config.dtype,
        required_names=required_names,
    )


def equal_interval_buckets(
    minimum: int,
    maximum: int,
    *,
    count: int,
    divisible_by: int = 1,
) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("bucket count must be positive")
    if minimum > maximum:
        raise ValueError("bucket minimum must be <= maximum")
    if divisible_by <= 0:
        raise ValueError("bucket divisible_by must be positive")
    if count == 1:
        return (int(maximum),)
    buckets = []
    span = maximum - minimum
    for index in range(count):
        if index == 0:
            value = minimum
        elif index == count - 1:
            value = maximum
        else:
            raw = minimum + span * index / (count - 1)
            value = int(round(raw))
            value = int(round(value / divisible_by) * divisible_by)
            value = min(max(value, minimum), maximum)
        if value % divisible_by != 0:
            raise ValueError(f"bucket value {value} is not divisible by {divisible_by}")
        buckets.append(int(value))
    return tuple(dict.fromkeys(buckets))


def float_input(values: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bfloat16":
        if values.dtype == np.uint16:
            return np.ascontiguousarray(values)
        return array_to_storage(values.astype(np.float32, copy=False), "bfloat16")
    return values.astype(dtype, copy=False)


def attach_profiling_metadata(spec: ModelSpec, shape_scenarios: list[dict[str, object]]) -> ModelSpec:
    spec.ir.setdefault("metadata", {})["profiling"] = {
        "version": 1,
        "shape_scenarios": shape_scenarios,
    }
    return spec


def enable_flash_attention_bias_for_target(config, *, target: str | None, needs_attention_mask: bool):
    if not needs_attention_mask or str(target).lower() != "rocm":
        return config
    text_config = getattr(config, "text_config", None)
    if text_config is None or bool(getattr(text_config, "use_flash_attention_bias", False)):
        return config
    return replace(config, text_config=replace(text_config, use_flash_attention_bias=True))


def _default_checkpoint_path(snapshot: Path) -> Path:
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        return index_path
    return snapshot / "model.safetensors"
