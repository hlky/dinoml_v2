from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import numpy as np

import dinoml as dml
from dinoml.ir import ModelSpec
from dinoml.models.clip.clip import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    clip_required_text_weight_names,
)
from dinoml.models.clip.clip_model import _build_input_ids
from dinoml.models.clip.workflow_common import load_clip_config, load_clip_weights, resolve_snapshot_paths


class WorkflowSettings(NamedTuple):
    snapshot: Path
    config_path: Path
    checkpoint_path: Path
    batch: int
    seq_len: int
    dtype: str
    with_projection: bool
    use_flash_attention: bool


def build_config(
    *,
    snapshot: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    batch: int | None = None,
    seq_len: int | None = None,
    dtype: str = "float32",
    with_projection: bool = True,
    use_flash_attention: bool = False,
):
    del batch, seq_len, with_projection
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
    batch = kwargs.pop("batch", None)
    seq_len = kwargs.pop("seq_len", None)
    del batch, seq_len
    settings = _resolve_settings(**kwargs)
    config = build_config(**settings._asdict())
    return load_clip_weights(
        config=config,
        snapshot=settings.snapshot,
        config_path=settings.config_path,
        checkpoint_path=settings.checkpoint_path,
        required_names=clip_required_text_weight_names(
            config.text_config,
            with_projection=settings.with_projection,
        ),
    )


def build_spec(
    *,
    snapshot: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    batch: int | None = None,
    seq_len: int | None = None,
    dtype: str = "float32",
    with_projection: bool = True,
    use_flash_attention: bool = False,
) -> ModelSpec:
    settings = _resolve_settings(
        snapshot=snapshot,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        batch=batch,
        seq_len=seq_len,
        dtype=dtype,
        with_projection=with_projection,
        use_flash_attention=use_flash_attention,
    )
    config = build_config(**settings._asdict())
    model = (
        CLIPTextModelWithProjection(config.text_config, build_weights(**settings._asdict()))
        if settings.with_projection
        else CLIPTextModel(config.text_config, build_weights(**settings._asdict()))
    )
    inputs = {"input_ids": dml.TensorSpec([settings.batch, settings.seq_len], "int64")}
    if not settings.use_flash_attention:
        inputs["attention_mask"] = dml.TensorSpec([settings.batch, settings.seq_len], "bool")
    return dml.trace(
        model,
        inputs=inputs,
        name=f"clip_text_{'proj' if settings.with_projection else 'pool'}_b{settings.batch}_s{settings.seq_len}",
    )


def build_validation_inputs(
    *,
    snapshot: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    batch: int | None = None,
    seq_len: int | None = None,
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
        seq_len=seq_len,
        dtype=dtype,
        use_flash_attention=use_flash_attention,
    )
    config = build_config(**settings._asdict())
    inputs = {
        "input_ids": _build_input_ids(config.text_config, batch=settings.batch, seq_len=settings.seq_len),
    }
    if not settings.use_flash_attention:
        inputs["attention_mask"] = np.ones((settings.batch, settings.seq_len), dtype=np.bool_)
    return inputs


def _resolve_settings(
    *,
    snapshot: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    batch: int | None = None,
    seq_len: int | None = None,
    dtype: str = "float32",
    with_projection: bool = True,
    use_flash_attention: bool = False,
) -> WorkflowSettings:
    resolved_snapshot, resolved_config_path, resolved_checkpoint_path = resolve_snapshot_paths(
        snapshot=snapshot,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    preview_config = load_clip_config(
        snapshot=resolved_snapshot,
        config_path=resolved_config_path,
        checkpoint_path=resolved_checkpoint_path,
        dtype=str(dtype),
        use_flash_attention=bool(use_flash_attention),
    )
    resolved_batch = 1 if batch is None else int(batch)
    resolved_seq_len = (
        min(int(preview_config.text_config.max_position_embeddings), 16) if seq_len is None else int(seq_len)
    )
    if resolved_batch <= 0:
        raise ValueError("batch must be positive")
    return WorkflowSettings(
        snapshot=resolved_snapshot,
        config_path=resolved_config_path,
        checkpoint_path=resolved_checkpoint_path,
        batch=resolved_batch,
        seq_len=resolved_seq_len,
        dtype=str(dtype),
        with_projection=bool(with_projection),
        use_flash_attention=bool(use_flash_attention),
    )
