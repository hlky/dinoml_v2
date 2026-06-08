from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from .clip import (
    clip_config_from_transformers_dict,
    clip_weights_from_safetensors_file,
    clip_weights_from_torch_file,
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


def load_clip_config(
    *,
    snapshot: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    dtype: str = "float32",
    use_flash_attention: bool = False,
):
    del checkpoint_path
    _, resolved_config_path, _ = resolve_snapshot_paths(snapshot=snapshot, config_path=config_path)
    payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    return clip_config_from_transformers_dict(
        payload,
        dtype=str(dtype),
        use_flash_attention=bool(use_flash_attention),
    )


def load_clip_weights(
    *,
    config,
    snapshot: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    required_names: Sequence[str] | None = None,
):
    del config_path
    _, _, resolved_checkpoint_path = resolve_snapshot_paths(snapshot=snapshot, checkpoint_path=checkpoint_path)
    if resolved_checkpoint_path.suffix == ".safetensors":
        return clip_weights_from_safetensors_file(
            resolved_checkpoint_path,
            config,
            dtype=config.dtype,
            required_names=required_names,
        )
    return clip_weights_from_torch_file(
        resolved_checkpoint_path,
        config,
        dtype=config.dtype,
        required_names=required_names,
    )


def float_input(values: np.ndarray, dtype: str) -> np.ndarray:
    return values.astype(dtype, copy=False)


def _default_checkpoint_path(snapshot: Path) -> Path:
    safetensors_path = snapshot / "model.safetensors"
    if safetensors_path.is_file():
        return safetensors_path
    return snapshot / "pytorch_model.bin"
