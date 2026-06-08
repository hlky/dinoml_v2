from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np

import dinoml as dml
from dinoml.ir import ModelSpec
from dinoml.models.qwen2_5_vl import (
    Qwen2_5_VLForConditionalGenerationImagePrefillWithCache,
    qwen2_5_vl_rope_index,
    qwen2_5_vl_text_rope_embeddings,
    qwen2_5_vl_vision_cu_seqlens,
    qwen2_5_vl_vision_position_ids,
    qwen2_5_vl_vision_rope_embeddings,
    qwen2_5_vl_vision_window_index,
)
from dinoml.models.qwen2_5_vl.workflow_common import (
    attach_profiling_metadata,
    enable_flash_attention_bias_for_target,
    equal_interval_buckets as _equal_interval_buckets,
    float_input as _float_input,
    load_qwen2_5_vl_config,
    load_qwen2_5_vl_weights,
    resolve_snapshot_paths,
)
from dinoml.shapes import Dim


class WorkflowSettings(NamedTuple):
    snapshot: Path
    config_path: Path
    checkpoint_path: Path
    grid_thw: tuple[int, int, int]
    min_grid_thw: tuple[int, int, int] | None
    max_grid_thw: tuple[int, int, int]
    prompt_len: int
    min_prompt_len: int
    max_prompt_len: int
    dtype: str
    grid_bucket_count: int
    prompt_bucket_count: int


def build_config(
    *,
    snapshot: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    grid_thw: str | Sequence[int] | None = None,
    min_grid_thw: str | Sequence[int] | None = None,
    max_grid_thw: str | Sequence[int] | None = None,
    prompt_len: int | None = None,
    min_prompt_len: int = 1,
    max_prompt_len: int | None = None,
    dtype: str = "bfloat16",
    grid_bucket_count: int = 3,
    prompt_bucket_count: int = 3,
    target: str | None = None,
):
    del grid_thw, min_grid_thw, max_grid_thw, prompt_len, min_prompt_len, max_prompt_len
    del grid_bucket_count, prompt_bucket_count
    settings = _resolve_settings(snapshot=snapshot, config_path=config_path, checkpoint_path=checkpoint_path, dtype=dtype)
    config = load_qwen2_5_vl_config(
        snapshot=settings.snapshot,
        config_path=settings.config_path,
        checkpoint_path=settings.checkpoint_path,
        dtype=settings.dtype,
    )
    return enable_flash_attention_bias_for_target(config, target=target, needs_attention_mask=True)


def build_weights(**kwargs):
    kwargs.pop("grid_bucket_count", None)
    kwargs.pop("prompt_bucket_count", None)
    target = kwargs.pop("target", None)
    settings = _resolve_settings(**kwargs)
    config = build_config(**kwargs, target=target)
    return load_qwen2_5_vl_weights(
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
    grid_thw: str | Sequence[int] | None = None,
    min_grid_thw: str | Sequence[int] | None = None,
    max_grid_thw: str | Sequence[int] | None = None,
    prompt_len: int | None = None,
    min_prompt_len: int = 1,
    max_prompt_len: int | None = None,
    dtype: str = "bfloat16",
    grid_bucket_count: int = 3,
    prompt_bucket_count: int = 3,
    target: str | None = None,
) -> ModelSpec:
    settings = _resolve_settings(
        snapshot=snapshot,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        grid_thw=grid_thw,
        min_grid_thw=min_grid_thw,
        max_grid_thw=max_grid_thw,
        prompt_len=prompt_len,
        min_prompt_len=min_prompt_len,
        max_prompt_len=max_prompt_len,
        dtype=dtype,
        grid_bucket_count=grid_bucket_count,
        prompt_bucket_count=prompt_bucket_count,
    )
    config = build_config(**settings._asdict(), target=target)
    merge = config.vision_config.spatial_merge_size
    _validate_dynamic_bounds(settings, merge)
    patch_count, image_feature_count, seq_len = _shape_dims(settings, merge)
    full_cu_count, window_cu_count = _cu_shape_dims(settings, config.vision_config, merge)
    inputs = {
        "input_ids": dml.TensorSpec([1, seq_len], "int32"),
        "pixel_values": dml.TensorSpec([patch_count, config.vision_config.patch_dim], config.vision_config.dtype),
        "image_grid_thw": dml.TensorSpec([1, 3], "float32"),
        "vision_cos": dml.TensorSpec([patch_count, config.vision_config.head_dim], "float32"),
        "vision_sin": dml.TensorSpec([patch_count, config.vision_config.head_dim], "float32"),
        "vision_full_cu_seqlens": dml.TensorSpec([full_cu_count], "int32"),
        "vision_window_cu_seqlens": dml.TensorSpec([window_cu_count], "int32"),
        "vision_reverse_window_index": dml.TensorSpec([image_feature_count], "int32"),
        "text_cos": dml.TensorSpec([1, seq_len, config.text_config.head_dim], config.text_config.dtype),
        "text_sin": dml.TensorSpec([1, seq_len, config.text_config.head_dim], config.text_config.dtype),
        "attention_mask": dml.TensorSpec([config.text_config.num_attention_heads, seq_len, seq_len], config.text_config.dtype),
    }
    spec = dml.trace(
        Qwen2_5_VLForConditionalGenerationImagePrefillWithCache(
            config,
            build_weights(**settings._asdict(), target=target),
            logits_to_keep=1,
            max_full_seqlen=_max_full_vision_seqlen(settings.max_grid_thw),
            max_window_seqlen=_max_window_vision_seqlen(config.vision_config),
        ),
        inputs=inputs,
        name=f"qwen2_5_vl_image_prefill_with_cache_grid{settings.max_grid_thw[0]}x{settings.max_grid_thw[1]}x{settings.max_grid_thw[2]}",
    )
    return attach_profiling_metadata(spec, _profiling_shape_scenarios(settings, config))


def build_validation_inputs(
    *,
    snapshot: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    grid_thw: str | Sequence[int] | None = None,
    min_grid_thw: str | Sequence[int] | None = None,
    max_grid_thw: str | Sequence[int] | None = None,
    prompt_len: int | None = None,
    min_prompt_len: int = 1,
    max_prompt_len: int | None = None,
    dtype: str = "bfloat16",
    grid_bucket_count: int = 3,
    prompt_bucket_count: int = 3,
    target: str | None = None,
) -> dict[str, np.ndarray]:
    del grid_bucket_count, prompt_bucket_count
    settings = _resolve_settings(
        snapshot=snapshot,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        grid_thw=grid_thw,
        min_grid_thw=min_grid_thw,
        max_grid_thw=max_grid_thw,
        prompt_len=prompt_len,
        min_prompt_len=min_prompt_len,
        max_prompt_len=max_prompt_len,
        dtype=dtype,
    )
    config = build_config(**settings._asdict(), target=target)
    image_grid_thw = np.asarray([settings.grid_thw], dtype=np.int32)
    image_feature_count = _image_feature_count(settings.grid_thw, config.vision_config.spatial_merge_size)
    patch_count = int(np.prod(settings.grid_thw))
    seq_len = 1 + image_feature_count + settings.prompt_len
    input_ids = np.concatenate(
        [
            np.asarray([42], dtype=np.int32),
            np.full((image_feature_count,), config.image_token_id, dtype=np.int32),
            np.full((settings.prompt_len,), 43, dtype=np.int32),
        ]
    ).reshape(1, seq_len)
    mm_token_type_ids = np.concatenate(
        [
            np.zeros((1,), dtype=np.int64),
            np.ones((image_feature_count,), dtype=np.int64),
            np.zeros((settings.prompt_len,), dtype=np.int64),
        ]
    ).reshape(1, seq_len)
    text_position_ids, _ = qwen2_5_vl_rope_index(
        input_ids,
        mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=config.vision_config.spatial_merge_size,
    )
    text_cos, text_sin = qwen2_5_vl_text_rope_embeddings(text_position_ids, config.text_config, dtype=config.text_config.dtype)
    vision_position_ids = qwen2_5_vl_vision_position_ids(image_grid_thw, config.vision_config.spatial_merge_size)
    vision_cos, vision_sin = qwen2_5_vl_vision_rope_embeddings(
        vision_position_ids,
        head_dim=config.vision_config.head_dim,
    )
    window_index, window_cu = qwen2_5_vl_vision_window_index(
        image_grid_thw,
        spatial_merge_size=config.vision_config.spatial_merge_size,
        window_size=config.vision_config.window_size,
        patch_size=config.vision_config.patch_size,
    )
    reverse_window_index = np.argsort(window_index).astype(np.int32)
    full_cu = qwen2_5_vl_vision_cu_seqlens(image_grid_thw).astype(np.int32)
    rng = np.random.default_rng(20260607)
    pixel_values = rng.normal(0.0, 0.2, (patch_count, config.vision_config.patch_dim)).astype(np.float32)
    pixel_values = _window_order_merged(pixel_values, window_index, config.vision_config.spatial_merge_unit)
    vision_cos = _window_order_merged(vision_cos, window_index, config.vision_config.spatial_merge_unit)
    vision_sin = _window_order_merged(vision_sin, window_index, config.vision_config.spatial_merge_unit)
    attention_mask = np.triu(
        np.full((config.text_config.num_attention_heads, seq_len, seq_len), -1.0e4, dtype=np.float32),
        k=1,
    )
    return {
        "input_ids": input_ids,
        "pixel_values": _float_input(pixel_values, config.vision_config.dtype),
        "image_grid_thw": image_grid_thw.astype(np.float32),
        "vision_cos": _float_input(vision_cos, "float32"),
        "vision_sin": _float_input(vision_sin, "float32"),
        "vision_full_cu_seqlens": full_cu,
        "vision_window_cu_seqlens": window_cu.astype(np.int32),
        "vision_reverse_window_index": reverse_window_index,
        "text_cos": _float_input(text_cos, config.text_config.dtype),
        "text_sin": _float_input(text_sin, config.text_config.dtype),
        "attention_mask": _float_input(attention_mask, config.text_config.dtype),
    }


def _parse_grid_thw(value: str | Sequence[int], *, name: str = "grid_thw") -> tuple[int, int, int]:
    if isinstance(value, str):
        parts = tuple(int(part) for part in value.split(","))
    else:
        parts = tuple(int(part) for part in value)
    if len(parts) != 3:
        raise ValueError(f"Expected {name} as three integers, got {value!r}")
    return parts


def _shape_dims(settings: WorkflowSettings, merge: int):
    min_grid_thw = _effective_min_grid_thw(settings, merge)
    patch_count_value = int(np.prod(settings.grid_thw))
    min_patch_count = int(np.prod(min_grid_thw))
    max_patch_count = int(np.prod(settings.max_grid_thw))
    image_feature_count_value = _image_feature_count(settings.grid_thw, merge)
    min_image_feature_count = _image_feature_count(min_grid_thw, merge)
    max_image_feature_count = _image_feature_count(settings.max_grid_thw, merge)
    seq_len_value = 1 + _image_feature_count(settings.grid_thw, merge) + settings.prompt_len
    min_seq_len = 1 + _image_feature_count(min_grid_thw, merge) + settings.min_prompt_len
    max_seq_len = 1 + _image_feature_count(settings.max_grid_thw, merge) + settings.max_prompt_len
    if min_patch_count == patch_count_value == max_patch_count and min_seq_len == seq_len_value == max_seq_len:
        return patch_count_value, image_feature_count_value, seq_len_value
    patch_count = Dim(
        "patch_count",
        min=min_patch_count,
        max=max_patch_count,
        typical=patch_count_value,
        buckets=_dim_buckets(min_patch_count, patch_count_value, max_patch_count),
    )
    image_feature_count = Dim(
        "image_feature_count",
        min=min_image_feature_count,
        max=max_image_feature_count,
        typical=image_feature_count_value,
        buckets=_dim_buckets(min_image_feature_count, image_feature_count_value, max_image_feature_count),
    )
    seq_len = Dim(
        "seq_len",
        min=min_seq_len,
        max=max_seq_len,
        typical=seq_len_value,
        buckets=_dim_buckets(
            min_seq_len,
            seq_len_value,
            max_seq_len,
        ),
    )
    return patch_count, image_feature_count, seq_len


def _profiling_shape_scenarios(settings: WorkflowSettings, config) -> list[dict[str, object]]:
    merge = int(config.vision_config.spatial_merge_size)
    min_grid_thw = _effective_min_grid_thw(settings, merge)
    if min_grid_thw[1] != min_grid_thw[2] or settings.max_grid_thw[1] != settings.max_grid_thw[2]:
        raise ValueError(
            "Qwen2.5-VL square profiling buckets require matching min/max grid height and width bounds"
        )
    grid_hw_values = _equal_interval_buckets(
        min_grid_thw[1],
        settings.max_grid_thw[1],
        count=settings.grid_bucket_count,
        divisible_by=merge,
    )
    prompt_values = _equal_interval_buckets(settings.min_prompt_len, settings.max_prompt_len, count=settings.prompt_bucket_count)
    scenarios = []
    for grid_hw in grid_hw_values:
        grid = (int(settings.grid_thw[0]), int(grid_hw), int(grid_hw))
        patch_count = int(np.prod(grid))
        image_feature_count = _image_feature_count(grid, merge)
        for prompt_len in prompt_values:
            seq_len = 1 + image_feature_count + int(prompt_len)
            scenarios.append(
                {
                    "source": "workflow_equal_interval_buckets",
                    "case_id": f"bucket_grid_t={grid[0]}_grid_h={grid[1]}_grid_w={grid[2]}_prompt_len={int(prompt_len)}",
                    "dim_values": {
                        "grid_t": grid[0],
                        "grid_h": grid[1],
                        "grid_w": grid[2],
                        "prompt_len": int(prompt_len),
                        "patch_count": patch_count,
                        "seq_len": seq_len,
                    },
                    "overrides": {
                        "input_ids": [1, seq_len],
                        "pixel_values": [patch_count, int(config.vision_config.patch_dim)],
                        "image_grid_thw": [1, 3],
                        "vision_cos": [patch_count, int(config.vision_config.head_dim)],
                        "vision_sin": [patch_count, int(config.vision_config.head_dim)],
                        "vision_full_cu_seqlens": [_full_cu_count(grid)],
                        "vision_window_cu_seqlens": [_window_cu_count(grid, config.vision_config)],
                        "vision_reverse_window_index": [image_feature_count],
                        "text_cos": [1, seq_len, int(config.text_config.head_dim)],
                        "text_sin": [1, seq_len, int(config.text_config.head_dim)],
                        "attention_mask": [int(config.text_config.num_attention_heads), seq_len, seq_len],
                    },
                }
            )
    return scenarios


def _image_feature_count(grid_thw: tuple[int, int, int], merge: int) -> int:
    grid_t, grid_h, grid_w = grid_thw
    if grid_h % merge or grid_w % merge:
        raise ValueError("Qwen2.5-VL image grid height and width must be divisible by spatial_merge_size")
    return int(grid_t * grid_h * grid_w // (merge * merge))


def _full_cu_count(grid_thw: tuple[int, int, int]) -> int:
    return int(grid_thw[0]) + 1


def _cu_shape_dims(settings: WorkflowSettings, vision_config, merge: int):
    min_grid_thw = _effective_min_grid_thw(settings, merge)
    full_counts = (
        _full_cu_count(min_grid_thw),
        _full_cu_count(settings.grid_thw),
        _full_cu_count(settings.max_grid_thw),
    )
    window_counts = (
        _window_cu_count(min_grid_thw, vision_config),
        _window_cu_count(settings.grid_thw, vision_config),
        _window_cu_count(settings.max_grid_thw, vision_config),
    )
    full_dim = _maybe_dim("vision_full_cu_count", full_counts)
    window_dim = _maybe_dim("vision_window_cu_count", window_counts)
    return full_dim, window_dim


def _maybe_dim(name: str, values: tuple[int, int, int]):
    minimum, typical, maximum = (int(value) for value in values)
    if minimum == typical == maximum:
        return typical
    return Dim(
        name,
        min=minimum,
        max=maximum,
        typical=typical,
        buckets=_dim_buckets(minimum, typical, maximum),
    )


def _window_cu_count(grid_thw: tuple[int, int, int], vision_config) -> int:
    _, cu = qwen2_5_vl_vision_window_index(
        np.asarray([grid_thw], dtype=np.int32),
        spatial_merge_size=int(vision_config.spatial_merge_size),
        window_size=int(vision_config.window_size),
        patch_size=int(vision_config.patch_size),
    )
    return int(cu.shape[0])


def _max_full_vision_seqlen(grid_thw: tuple[int, int, int]) -> int:
    return int(grid_thw[1] * grid_thw[2])


def _max_window_vision_seqlen(vision_config) -> int:
    return int(vision_config.spatial_merge_unit * vision_config.vit_merger_window_size * vision_config.vit_merger_window_size)


def _window_order_merged(values: np.ndarray, window_index: np.ndarray, merge_unit: int) -> np.ndarray:
    value = np.asarray(values)
    grouped = value.reshape(-1, int(merge_unit), value.shape[-1])
    return grouped[np.asarray(window_index, dtype=np.int64)].reshape(value.shape)


def _validate_dynamic_bounds(settings: WorkflowSettings, merge: int) -> None:
    min_grid_thw = _effective_min_grid_thw(settings, merge)
    _image_feature_count(settings.grid_thw, merge)
    _image_feature_count(min_grid_thw, merge)
    _image_feature_count(settings.max_grid_thw, merge)
    if any(minimum > current for minimum, current in zip(min_grid_thw, settings.grid_thw, strict=True)):
        raise ValueError("Qwen2.5-VL min_grid_thw must be <= grid_thw elementwise")
    if any(current > maximum for current, maximum in zip(settings.grid_thw, settings.max_grid_thw, strict=True)):
        raise ValueError("Qwen2.5-VL grid_thw must be <= max_grid_thw elementwise")
    if settings.min_prompt_len > settings.prompt_len or settings.prompt_len > settings.max_prompt_len:
        raise ValueError("expected min_prompt_len <= prompt_len <= max_prompt_len")


def _resolve_settings(
    *,
    snapshot: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    grid_thw: str | Sequence[int] | None = None,
    min_grid_thw: str | Sequence[int] | None = None,
    max_grid_thw: str | Sequence[int] | None = None,
    prompt_len: int | None = None,
    min_prompt_len: int = 1,
    max_prompt_len: int | None = None,
    dtype: str = "bfloat16",
    grid_bucket_count: int = 3,
    prompt_bucket_count: int = 3,
) -> WorkflowSettings:
    resolved_snapshot, resolved_config_path, resolved_checkpoint_path = resolve_snapshot_paths(
        snapshot=snapshot,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    resolved_min_grid_thw = None if min_grid_thw is None else _parse_grid_thw(min_grid_thw, name="min_grid_thw")
    resolved_grid_thw = resolved_min_grid_thw if grid_thw is None and resolved_min_grid_thw is not None else _parse_grid_thw(grid_thw or "1,8,8")
    default_max_grid_thw = (resolved_grid_thw[0], max(resolved_grid_thw[1], 16), max(resolved_grid_thw[2], 16))
    resolved_max_grid_thw = default_max_grid_thw if max_grid_thw is None else _parse_grid_thw(max_grid_thw, name="max_grid_thw")
    resolved_min_prompt_len = int(min_prompt_len)
    resolved_prompt_len = resolved_min_prompt_len if prompt_len is None else int(prompt_len)
    resolved_max_prompt_len = max(resolved_prompt_len, 16) if max_prompt_len is None else int(max_prompt_len)
    return WorkflowSettings(
        snapshot=resolved_snapshot,
        config_path=resolved_config_path,
        checkpoint_path=resolved_checkpoint_path,
        grid_thw=resolved_grid_thw,
        min_grid_thw=resolved_min_grid_thw,
        max_grid_thw=resolved_max_grid_thw,
        prompt_len=resolved_prompt_len,
        min_prompt_len=resolved_min_prompt_len,
        max_prompt_len=resolved_max_prompt_len,
        dtype=str(dtype),
        grid_bucket_count=int(grid_bucket_count),
        prompt_bucket_count=int(prompt_bucket_count),
    )


def _effective_min_grid_thw(settings: WorkflowSettings, merge: int) -> tuple[int, int, int]:
    if settings.min_grid_thw is not None:
        return settings.min_grid_thw
    return (1, merge, merge)


def _dim_buckets(*values: int) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(value) for value in values))
