from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import numpy as np

import dinoml as dml
from dinoml.ir import ModelSpec
from dinoml.models.clip.clip import (
    CLIPVisionModel,
    CLIPVisionModelWithProjection,
    clip_required_vision_weight_names,
)
from dinoml.models.clip.clip_model import _build_pixel_values
from dinoml.models.clip.workflow_common import load_clip_config, load_clip_weights, resolve_snapshot_paths


class WorkflowSettings(NamedTuple):
    snapshot: Path
    config_path: Path
    checkpoint_path: Path
    batch: int
    dtype: str
    with_projection: bool
    use_flash_attention: bool


def build_config(
    *,
    snapshot: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    batch: int | None = None,
    dtype: str = "float32",
    with_projection: bool = True,
    use_flash_attention: bool = False,
):
    del batch, with_projection
    settings = _resolve_settings(
        snapshot=snapshot,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        dtype=dtype,
        use_flash_attention=use_flash_attention,
    )
    return load_clip_config(
        snapshot=settings.snapshot,
        config_path=settings.config_path,
        checkpoint_path=settings.checkpoint_path,
        dtype=settings.dtype,
        use_flash_attention=settings.use_flash_attention,
    )


def build_weights(**kwargs):
    kwargs.pop("batch", None)
    settings = _resolve_settings(**kwargs)
    config = build_config(**settings._asdict())
    return load_clip_weights(
        config=config,
        snapshot=settings.snapshot,
        config_path=settings.config_path,
        checkpoint_path=settings.checkpoint_path,
        required_names=clip_required_vision_weight_names(
            config.vision_config,
            with_projection=settings.with_projection,
        ),
    )


def build_spec(
    *,
    snapshot: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    batch: int | None = None,
    dtype: str = "float32",
    with_projection: bool = True,
    use_flash_attention: bool = False,
) -> ModelSpec:
    settings = _resolve_settings(
        snapshot=snapshot,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        batch=batch,
        dtype=dtype,
        with_projection=with_projection,
        use_flash_attention=use_flash_attention,
    )
    config = build_config(**settings._asdict())
    model = (
        CLIPVisionModelWithProjection(config.vision_config, build_weights(**settings._asdict()))
        if settings.with_projection
        else CLIPVisionModel(config.vision_config, build_weights(**settings._asdict()))
    )
    return dml.trace(
        model,
        inputs={
            "pixel_values": dml.TensorSpec(
                [
                    settings.batch,
                    int(config.vision_config.num_channels),
                    int(config.vision_config.image_size),
                    int(config.vision_config.image_size),
                ],
                config.vision_config.dtype,
            )
        },
        name=f"clip_vision_{'proj' if settings.with_projection else 'pool'}_b{settings.batch}",
    )


def build_validation_inputs(
    *,
    snapshot: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    batch: int | None = None,
    dtype: str = "float32",
    with_projection: bool = True,
    use_flash_attention: bool = False,
) -> dict[str, np.ndarray]:
    del with_projection
    settings = _resolve_settings(
        snapshot=snapshot,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        batch=batch,
        dtype=dtype,
        use_flash_attention=use_flash_attention,
    )
    config = build_config(**settings._asdict())
    return {
        "pixel_values": _build_pixel_values(config.vision_config, batch=settings.batch, dtype=settings.dtype),
    }


def _resolve_settings(
    *,
    snapshot: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    batch: int | None = None,
    dtype: str = "float32",
    with_projection: bool = True,
    use_flash_attention: bool = False,
) -> WorkflowSettings:
    resolved_snapshot, resolved_config_path, resolved_checkpoint_path = resolve_snapshot_paths(
        snapshot=snapshot,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    resolved_batch = 1 if batch is None else int(batch)
    if resolved_batch <= 0:
        raise ValueError("batch must be positive")
    return WorkflowSettings(
        snapshot=resolved_snapshot,
        config_path=resolved_config_path,
        checkpoint_path=resolved_checkpoint_path,
        batch=resolved_batch,
        dtype=str(dtype),
        with_projection=bool(with_projection),
        use_flash_attention=bool(use_flash_attention),
    )
