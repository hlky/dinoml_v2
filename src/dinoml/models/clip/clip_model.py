from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import numpy as np

import dinoml as dml
from dinoml.ir import ModelSpec
from dinoml.models.clip.clip import CLIPModel
from dinoml.models.clip.workflow_common import float_input as _float_input
from dinoml.models.clip.workflow_common import load_clip_config, load_clip_weights, resolve_snapshot_paths


class WorkflowSettings(NamedTuple):
    snapshot: Path
    config_path: Path
    checkpoint_path: Path
    text_batch: int
    image_batch: int
    seq_len: int
    dtype: str
    use_flash_attention: bool


def build_config(
    *,
    snapshot: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    text_batch: int | None = None,
    image_batch: int | None = None,
    seq_len: int | None = None,
    dtype: str = "float32",
    use_flash_attention: bool = False,
):
    del text_batch, image_batch, seq_len
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
    for key in ("text_batch", "image_batch", "seq_len"):
        kwargs.pop(key, None)
    settings = _resolve_settings(**kwargs)
    config = build_config(**settings._asdict())
    return load_clip_weights(
        config=config,
        snapshot=settings.snapshot,
        config_path=settings.config_path,
        checkpoint_path=settings.checkpoint_path,
    )


def build_spec(
    *,
    snapshot: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    text_batch: int | None = None,
    image_batch: int | None = None,
    seq_len: int | None = None,
    dtype: str = "float32",
    use_flash_attention: bool = False,
) -> ModelSpec:
    settings = _resolve_settings(
        snapshot=snapshot,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        text_batch=text_batch,
        image_batch=image_batch,
        seq_len=seq_len,
        dtype=dtype,
        use_flash_attention=use_flash_attention,
    )
    config = build_config(**settings._asdict())
    inputs = {
        "input_ids": dml.TensorSpec([settings.text_batch, settings.seq_len], "int64"),
        "pixel_values": dml.TensorSpec(
            [
                settings.image_batch,
                int(config.vision_config.num_channels),
                int(config.vision_config.image_size),
                int(config.vision_config.image_size),
            ],
            config.vision_config.dtype,
        ),
    }
    if not settings.use_flash_attention:
        inputs["attention_mask"] = dml.TensorSpec([settings.text_batch, settings.seq_len], "bool")
    return dml.trace(
        CLIPModel(config, build_weights(**settings._asdict())),
        inputs=inputs,
        name=f"clip_model_tb{settings.text_batch}_ib{settings.image_batch}_s{settings.seq_len}",
    )


def build_validation_inputs(
    *,
    snapshot: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    text_batch: int | None = None,
    image_batch: int | None = None,
    seq_len: int | None = None,
    dtype: str = "float32",
    use_flash_attention: bool = False,
) -> dict[str, np.ndarray]:
    settings = _resolve_settings(
        snapshot=snapshot,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        text_batch=text_batch,
        image_batch=image_batch,
        seq_len=seq_len,
        dtype=dtype,
        use_flash_attention=use_flash_attention,
    )
    config = build_config(**settings._asdict())
    inputs = {
        "input_ids": _build_input_ids(config.text_config, batch=settings.text_batch, seq_len=settings.seq_len),
        "pixel_values": _build_pixel_values(
            config.vision_config,
            batch=settings.image_batch,
            dtype=settings.dtype,
        ),
    }
    if not settings.use_flash_attention:
        inputs["attention_mask"] = np.ones((settings.text_batch, settings.seq_len), dtype=np.bool_)
    return inputs


def _build_input_ids(text_config, *, batch: int, seq_len: int) -> np.ndarray:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if seq_len > int(text_config.max_position_embeddings):
        raise ValueError("seq_len must be <= max_position_embeddings")
    eos_token_id = int(text_config.eos_token_id)
    vocab_size = int(text_config.vocab_size)
    row = []
    candidate = 0
    while len(row) < max(seq_len - 1, 0):
        value = candidate % vocab_size
        if value != eos_token_id:
            row.append(value)
        candidate += 1
    row.append(vocab_size - 1 if eos_token_id == 2 else eos_token_id)
    return np.tile(np.asarray(row, dtype=np.int64), (batch, 1))


def _build_pixel_values(vision_config, *, batch: int, dtype: str) -> np.ndarray:
    image_size = int(vision_config.image_size)
    num_channels = int(vision_config.num_channels)
    values = np.linspace(
        -1.0,
        1.0,
        num=batch * num_channels * image_size * image_size,
        dtype=np.float32,
    ).reshape(batch, num_channels, image_size, image_size)
    return _float_input(values, dtype)


def _resolve_settings(
    *,
    snapshot: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    text_batch: int | None = None,
    image_batch: int | None = None,
    seq_len: int | None = None,
    dtype: str = "float32",
    use_flash_attention: bool = False,
) -> WorkflowSettings:
    resolved_snapshot, resolved_config_path, resolved_checkpoint_path = resolve_snapshot_paths(
        snapshot=snapshot,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    resolved_text_batch = 1 if text_batch is None else int(text_batch)
    resolved_image_batch = 1 if image_batch is None else int(image_batch)
    preview_config = load_clip_config(
        snapshot=resolved_snapshot,
        config_path=resolved_config_path,
        checkpoint_path=resolved_checkpoint_path,
        dtype=str(dtype),
        use_flash_attention=bool(use_flash_attention),
    )
    default_seq_len = min(int(preview_config.text_config.max_position_embeddings), 16)
    resolved_seq_len = default_seq_len if seq_len is None else int(seq_len)
    if resolved_text_batch <= 0 or resolved_image_batch <= 0:
        raise ValueError("text_batch and image_batch must be positive")
    return WorkflowSettings(
        snapshot=resolved_snapshot,
        config_path=resolved_config_path,
        checkpoint_path=resolved_checkpoint_path,
        text_batch=resolved_text_batch,
        image_batch=resolved_image_batch,
        seq_len=resolved_seq_len,
        dtype=str(dtype),
        use_flash_attention=bool(use_flash_attention),
    )
