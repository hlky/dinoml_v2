from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np

import dinoml as dml
from dinoml.ir import ModelSpec
from dinoml.models.glm_ocr import (
    GlmOcrForConditionalGenerationImagePrefill,
    GlmOcrForConditionalGenerationImagePrefillWithCache,
    glm_ocr_rope_index,
    glm_ocr_text_rope_embeddings,
    glm_ocr_vision_position_ids,
    glm_ocr_vision_rope_embeddings,
)
from dinoml.shapes import Dim
from dinoml.models.glm_ocr.workflow_common import (
    attach_profiling_metadata,
    equal_interval_buckets as _equal_interval_buckets,
    float_input as _float_input,
    load_glm_ocr_config,
    load_glm_ocr_weights,
    resolve_snapshot_paths,
)


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


def _parse_grid_thw(value: str | Sequence[int], *, name: str = "grid_thw") -> tuple[int, int, int]:
    if isinstance(value, str):
        parts = tuple(int(part) for part in value.split(","))
    else:
        parts = tuple(int(part) for part in value)
    if len(parts) != 3:
        raise ValueError(f"Expected {name} as three integers, got {value!r}")
    return parts


def _dim_buckets(*values: int) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(value) for value in values))


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
    compile_cache: bool = False,
):
    del grid_bucket_count, prompt_bucket_count, compile_cache
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
    return load_glm_ocr_config(
        snapshot=settings.snapshot,
        config_path=settings.config_path,
        checkpoint_path=settings.checkpoint_path,
        dtype=settings.dtype,
    )


def build_weights(**kwargs):
    kwargs.pop("grid_bucket_count", None)
    kwargs.pop("prompt_bucket_count", None)
    kwargs.pop("compile_cache", None)
    settings = _resolve_settings(**kwargs)
    config = build_config(**kwargs)
    return load_glm_ocr_weights(
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
    compile_cache: bool = False,
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
    config = build_config(**settings._asdict())
    merge = config.vision_config.spatial_merge_size
    _validate_dynamic_bounds(settings, merge)
    patch_count, seq_len = _dynamic_shape_dims(settings, merge)
    max_seq_len = 1 + _image_feature_count(settings.max_grid_thw, merge) + settings.max_prompt_len
    inputs = {
        "input_ids": dml.TensorSpec([1, seq_len], "int64"),
        "pixel_values": dml.TensorSpec([patch_count, config.vision_config.patch_dim], config.vision_config.dtype),
        "vision_cos": dml.TensorSpec([patch_count, config.vision_config.head_dim], config.vision_config.dtype),
        "vision_sin": dml.TensorSpec([patch_count, config.vision_config.head_dim], config.vision_config.dtype),
        "text_cos": dml.TensorSpec([1, seq_len, config.text_config.head_dim], config.text_config.dtype),
        "text_sin": dml.TensorSpec([1, seq_len, config.text_config.head_dim], config.text_config.dtype),
    }
    model_cls = GlmOcrForConditionalGenerationImagePrefill
    spec_name = "glm_ocr_official_image_prefill_dynamic_shape"
    logits_to_keep = 0
    if compile_cache:
        inputs["attention_mask"] = dml.TensorSpec(
            [config.text_config.num_attention_heads, seq_len, seq_len],
            config.text_config.dtype,
        )
        model_cls = GlmOcrForConditionalGenerationImagePrefillWithCache
        spec_name = f"{spec_name}_with_cache"
        logits_to_keep = 1
    spec = dml.trace(
        model_cls(
            config,
            build_weights(**settings._asdict()),
            image_token_start=1,
            logits_to_keep=logits_to_keep,
        ),
        inputs=inputs,
        name=f"{spec_name}_grid{settings.max_grid_thw[0]}x{settings.max_grid_thw[1]}x{settings.max_grid_thw[2]}_s{max_seq_len}",
    )
    return attach_profiling_metadata(
        spec,
        _profiling_shape_scenarios(settings, config, compile_cache=compile_cache),
    )


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
    compile_cache: bool = False,
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
    config = build_config(**settings._asdict())
    image_grid_thw = np.asarray([settings.grid_thw], dtype=np.int64)
    image_feature_count = _image_feature_count(settings.grid_thw, config.vision_config.spatial_merge_size)
    patch_count = int(np.prod(settings.grid_thw))
    seq_len = 1 + image_feature_count + settings.prompt_len
    input_ids = np.concatenate(
        [
            np.asarray([42], dtype=np.int64),
            np.full((image_feature_count,), config.image_token_id, dtype=np.int64),
            np.full((settings.prompt_len,), 43, dtype=np.int64),
        ]
    ).reshape(1, seq_len)
    mm_token_type_ids = np.concatenate(
        [
            np.zeros((1,), dtype=np.int64),
            np.ones((image_feature_count,), dtype=np.int64),
            np.zeros((settings.prompt_len,), dtype=np.int64),
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
    inputs = {
        "input_ids": input_ids,
        "pixel_values": _float_input(pixel_values, config.vision_config.dtype),
        "vision_cos": _float_input(vision_cos, config.vision_config.dtype),
        "vision_sin": _float_input(vision_sin, config.vision_config.dtype),
        "text_cos": _float_input(text_cos, config.text_config.dtype),
        "text_sin": _float_input(text_sin, config.text_config.dtype),
    }
    if compile_cache:
        attention_mask = np.triu(
            np.full((config.text_config.num_attention_heads, seq_len, seq_len), -1.0e4, dtype=np.float32),
            k=1,
        )
        inputs["attention_mask"] = _float_input(attention_mask, config.text_config.dtype)
    return inputs


def _dynamic_shape_dims(settings: WorkflowSettings, merge: int):
    min_grid_thw = _effective_min_grid_thw(settings, merge)
    min_patch_count = int(np.prod(min_grid_thw))
    max_patch_count = int(np.prod(settings.max_grid_thw))
    typical_patch_count = int(np.prod(settings.grid_thw))
    min_seq_len = 1 + _image_feature_count(min_grid_thw, merge) + settings.min_prompt_len
    max_seq_len = 1 + _image_feature_count(settings.max_grid_thw, merge) + settings.max_prompt_len
    typical_seq_len = 1 + _image_feature_count(settings.grid_thw, merge) + settings.prompt_len
    patch_count = Dim(
        "patch_count",
        min=min_patch_count,
        max=max_patch_count,
        typical=typical_patch_count,
        buckets=_dim_buckets(min_patch_count, typical_patch_count, max_patch_count),
    )
    seq_len = Dim(
        "seq_len",
        min=min_seq_len,
        max=max_seq_len,
        typical=typical_seq_len,
        buckets=_dim_buckets(min_seq_len, typical_seq_len, max_seq_len),
    )
    return patch_count, seq_len


def _profiling_shape_scenarios(
    settings: WorkflowSettings,
    config,
    *,
    compile_cache: bool,
) -> list[dict[str, object]]:
    merge = int(config.vision_config.spatial_merge_size)
    min_grid_thw = _effective_min_grid_thw(settings, merge)
    if min_grid_thw[1] != min_grid_thw[2] or settings.max_grid_thw[1] != settings.max_grid_thw[2]:
        raise ValueError(
            "GLM_OCR square profiling buckets require matching min/max grid height and width bounds"
        )
    grid_t_values = _equal_interval_buckets(
        min_grid_thw[0],
        settings.max_grid_thw[0],
        count=settings.grid_bucket_count,
        divisible_by=1,
    )
    grid_hw_values = _equal_interval_buckets(
        min_grid_thw[1],
        settings.max_grid_thw[1],
        count=settings.grid_bucket_count,
        divisible_by=merge,
    )
    prompt_values = _equal_interval_buckets(
        settings.min_prompt_len,
        settings.max_prompt_len,
        count=settings.prompt_bucket_count,
        divisible_by=1,
    )
    scenarios = []
    for grid_t in grid_t_values:
        for grid_hw in grid_hw_values:
            grid_thw = (int(grid_t), int(grid_hw), int(grid_hw))
            patch_count = int(np.prod(grid_thw))
            image_feature_count = _image_feature_count(grid_thw, merge)
            for prompt_len in prompt_values:
                seq_len = 1 + image_feature_count + int(prompt_len)
                case_id = (
                    f"bucket_grid_t={grid_t}_grid_h={grid_hw}_grid_w={grid_hw}_prompt_len={int(prompt_len)}"
                )
                scenarios.append(
                    {
                        "source": "workflow_equal_interval_buckets",
                        "case_id": case_id,
                        "dim_values": {
                            "grid_t": int(grid_t),
                            "grid_h": int(grid_hw),
                            "grid_w": int(grid_hw),
                            "prompt_len": int(prompt_len),
                            "patch_count": patch_count,
                            "seq_len": seq_len,
                        },
                        "overrides": {
                            "input_ids": [1, seq_len],
                            "pixel_values": [patch_count, int(config.vision_config.patch_dim)],
                            "vision_cos": [patch_count, int(config.vision_config.head_dim)],
                            "vision_sin": [patch_count, int(config.vision_config.head_dim)],
                            "text_cos": [1, seq_len, int(config.text_config.head_dim)],
                            "text_sin": [1, seq_len, int(config.text_config.head_dim)],
                            **(
                                {
                                    "attention_mask": [
                                        int(config.text_config.num_attention_heads),
                                        seq_len,
                                        seq_len,
                                    ]
                                }
                                if compile_cache
                                else {}
                            ),
                        },
                    }
                )
    return scenarios


def _image_feature_count(grid_thw: tuple[int, int, int], merge: int) -> int:
    grid_t, grid_h, grid_w = grid_thw
    if grid_h % merge or grid_w % merge:
        raise ValueError("GLM_OCR image grid height and width must be divisible by spatial_merge_size")
    return int(grid_t * grid_h * grid_w // (merge * merge))


def _validate_dynamic_bounds(settings: WorkflowSettings, merge: int) -> None:
    min_grid_thw = _effective_min_grid_thw(settings, merge)
    if settings.grid_thw[0] <= 0 or min_grid_thw[0] <= 0 or settings.max_grid_thw[0] <= 0:
        raise ValueError("GLM_OCR image grid temporal dimension must be positive")
    if settings.prompt_len <= 0 or settings.min_prompt_len <= 0 or settings.max_prompt_len <= 0:
        raise ValueError("GLM_OCR prompt length must be positive")
    _image_feature_count(settings.grid_thw, merge)
    _image_feature_count(min_grid_thw, merge)
    _image_feature_count(settings.max_grid_thw, merge)
    if any(minimum > current for minimum, current in zip(min_grid_thw, settings.grid_thw, strict=True)):
        raise ValueError("GLM_OCR min_grid_thw must be <= grid_thw elementwise")
    if any(current > maximum for current, maximum in zip(settings.grid_thw, settings.max_grid_thw, strict=True)):
        raise ValueError("GLM_OCR grid_thw must be <= max_grid_thw elementwise")
    if settings.min_prompt_len > settings.prompt_len:
        raise ValueError("GLM_OCR min_prompt_len must be <= prompt_len")
    if settings.prompt_len > settings.max_prompt_len:
        raise ValueError("GLM_OCR prompt_len must be <= max_prompt_len")


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
    if grid_thw is None:
        if resolved_min_grid_thw is not None:
            resolved_grid_thw = resolved_min_grid_thw
        else:
            resolved_grid_thw = (1, 8, 8)
    else:
        resolved_grid_thw = _parse_grid_thw(grid_thw, name="grid_thw")
    default_max_grid_thw = (
        resolved_grid_thw[0],
        max(resolved_grid_thw[1], 16),
        max(resolved_grid_thw[2], 16),
    )
    resolved_max_grid_thw = default_max_grid_thw if max_grid_thw is None else _parse_grid_thw(max_grid_thw, name="max_grid_thw")
    resolved_min_prompt_len = int(min_prompt_len)
    resolved_prompt_len = resolved_min_prompt_len if prompt_len is None else int(prompt_len)
    default_max_prompt_len = max(resolved_prompt_len, 16)
    resolved_max_prompt_len = default_max_prompt_len if max_prompt_len is None else int(max_prompt_len)
    resolved_grid_bucket_count = int(grid_bucket_count)
    resolved_prompt_bucket_count = int(prompt_bucket_count)
    if resolved_grid_bucket_count <= 0:
        raise ValueError("grid_bucket_count must be positive")
    if resolved_prompt_bucket_count <= 0:
        raise ValueError("prompt_bucket_count must be positive")
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
        grid_bucket_count=resolved_grid_bucket_count,
        prompt_bucket_count=resolved_prompt_bucket_count,
    )


def _effective_min_grid_thw(settings: WorkflowSettings, merge: int) -> tuple[int, int, int]:
    if settings.min_grid_thw is not None:
        return settings.min_grid_thw
    return (1, merge, merge)
