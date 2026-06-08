from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import numpy as np

import dinoml as dml
from dinoml.ir import ModelSpec
from dinoml.models.qwen2_5_vl import (
    Qwen2_5_VLForConditionalGenerationDecode,
    Qwen2_5_VLForConditionalGenerationDecodeStaticCache,
    qwen2_5_vl_required_text_weight_names,
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
from dinoml.shapes import Dim, symbolic_int_expr


_CACHE_VARIANTS = frozenset({"none", "static"})


class WorkflowSettings(NamedTuple):
    snapshot: Path
    config_path: Path
    checkpoint_path: Path
    batch: int
    min_batch: int
    max_batch: int
    past_len: int
    min_past_len: int
    max_past_len: int
    dtype: str
    batch_bucket_count: int
    past_bucket_count: int


def build_config(
    *,
    snapshot: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    batch: int | None = None,
    min_batch: int = 1,
    max_batch: int | None = None,
    past_len: int | None = None,
    min_past_len: int = 1,
    max_past_len: int | None = None,
    dtype: str = "bfloat16",
    cache_variant: str = "none",
    batch_bucket_count: int = 3,
    past_bucket_count: int = 3,
    use_attention_mask: bool = False,
    target: str | None = None,
):
    del batch, min_batch, max_batch, past_len, min_past_len, max_past_len
    del cache_variant, batch_bucket_count, past_bucket_count
    settings = _resolve_settings(snapshot=snapshot, config_path=config_path, checkpoint_path=checkpoint_path, dtype=dtype)
    config = load_qwen2_5_vl_config(
        snapshot=settings.snapshot,
        config_path=settings.config_path,
        checkpoint_path=settings.checkpoint_path,
        dtype=settings.dtype,
    )
    return enable_flash_attention_bias_for_target(config, target=target, needs_attention_mask=use_attention_mask)


def build_weights(**kwargs):
    for key in ("batch", "min_batch", "max_batch", "past_len", "min_past_len", "max_past_len", "cache_variant", "batch_bucket_count", "past_bucket_count", "use_attention_mask"):
        kwargs.pop(key, None)
    target = kwargs.pop("target", None)
    settings = _resolve_settings(**kwargs)
    config = build_config(**kwargs, target=target)
    return load_qwen2_5_vl_weights(
        config=config,
        snapshot=settings.snapshot,
        config_path=settings.config_path,
        checkpoint_path=settings.checkpoint_path,
        required_names=qwen2_5_vl_required_text_weight_names(config),
    )


def build_spec(
    *,
    snapshot: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    batch: int | None = None,
    min_batch: int = 1,
    max_batch: int | None = None,
    past_len: int | None = None,
    min_past_len: int = 1,
    max_past_len: int | None = None,
    dtype: str = "bfloat16",
    cache_variant: str = "none",
    batch_bucket_count: int = 3,
    past_bucket_count: int = 3,
    use_attention_mask: bool = False,
    target: str | None = None,
) -> ModelSpec:
    settings = _resolve_settings(
        snapshot=snapshot,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        batch=batch,
        min_batch=min_batch,
        max_batch=max_batch,
        past_len=past_len,
        min_past_len=min_past_len,
        max_past_len=max_past_len,
        dtype=dtype,
        batch_bucket_count=batch_bucket_count,
        past_bucket_count=past_bucket_count,
    )
    variant = _normalize_cache_variant(cache_variant)
    config = build_config(**settings._asdict(), cache_variant=variant, use_attention_mask=use_attention_mask, target=target)
    weights = build_weights(**settings._asdict(), target=target)
    batch_dim = Dim("batch", min=settings.min_batch, max=settings.max_batch, typical=settings.batch, buckets=_equal_interval_buckets(settings.min_batch, settings.max_batch, count=settings.batch_bucket_count))
    past_len_dim = Dim("past_len", min=settings.min_past_len, max=settings.max_past_len, typical=settings.past_len, buckets=_equal_interval_buckets(settings.min_past_len, settings.max_past_len, count=settings.past_bucket_count))
    if variant == "static":
        cache_len_dim = Dim(
            "cache_len",
            min=settings.min_past_len + 1,
            max=settings.max_past_len + 1,
            typical=settings.past_len + 1,
            buckets=tuple(value + 1 for value in _equal_interval_buckets(settings.min_past_len, settings.max_past_len, count=settings.past_bucket_count)),
        )
        total_len_dim = cache_len_dim
    else:
        total_len_dim = symbolic_int_expr("add", past_len_dim.to_json(), 1)
    batch_heads_dim = symbolic_int_expr("mul", batch_dim.to_json(), config.text_config.num_attention_heads)
    inputs: dict[str, dml.TensorSpec] = {
        "input_ids": dml.TensorSpec([batch_dim, 1], "int32"),
        "cos": dml.TensorSpec([batch_dim, 1, config.text_config.head_dim], config.text_config.dtype),
        "sin": dml.TensorSpec([batch_dim, 1, config.text_config.head_dim], config.text_config.dtype),
    }
    if use_attention_mask:
        inputs["attention_mask"] = dml.TensorSpec([batch_heads_dim, 1, total_len_dim], config.text_config.dtype)
    if variant == "static":
        inputs["cache_seqlens"] = dml.TensorSpec([batch_dim], "int32")
    for layer_idx in range(config.text_config.num_hidden_layers):
        buffer_len = total_len_dim if variant == "static" else past_len_dim
        inputs[f"past_key_{layer_idx}"] = dml.TensorSpec([batch_dim, config.text_config.num_key_value_heads, buffer_len, config.text_config.head_dim], config.text_config.dtype)
        inputs[f"past_value_{layer_idx}"] = dml.TensorSpec([batch_dim, config.text_config.num_key_value_heads, buffer_len, config.text_config.head_dim], config.text_config.dtype)
    model = Qwen2_5_VLForConditionalGenerationDecodeStaticCache(config, weights) if variant == "static" else Qwen2_5_VLForConditionalGenerationDecode(config, weights)
    spec = dml.trace(model, inputs=inputs, name=f"qwen2_5_vl_decode_{variant}_b{settings.max_batch}_past{settings.max_past_len}")
    return attach_profiling_metadata(spec, _profiling_shape_scenarios(settings, config, cache_variant=variant, use_attention_mask=use_attention_mask))


def build_validation_inputs(
    *,
    snapshot: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    batch: int | None = None,
    min_batch: int = 1,
    max_batch: int | None = None,
    past_len: int | None = None,
    min_past_len: int = 1,
    max_past_len: int | None = None,
    dtype: str = "bfloat16",
    cache_variant: str = "none",
    batch_bucket_count: int = 3,
    past_bucket_count: int = 3,
    use_attention_mask: bool = False,
    target: str | None = None,
) -> dict[str, np.ndarray]:
    settings = _resolve_settings(
        snapshot=snapshot,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        batch=batch,
        min_batch=min_batch,
        max_batch=max_batch,
        past_len=past_len,
        min_past_len=min_past_len,
        max_past_len=max_past_len,
        dtype=dtype,
        batch_bucket_count=batch_bucket_count,
        past_bucket_count=past_bucket_count,
    )
    variant = _normalize_cache_variant(cache_variant)
    config = build_config(**settings._asdict(), cache_variant=variant, use_attention_mask=use_attention_mask, target=target)
    rng = np.random.default_rng(20260607)
    total_len = settings.past_len + 1
    inputs = {
        "input_ids": np.full((settings.batch, 1), 42, dtype=np.int32),
        "cos": _float_input(np.ones((settings.batch, 1, config.text_config.head_dim), dtype=np.float32), config.text_config.dtype),
        "sin": _float_input(np.zeros((settings.batch, 1, config.text_config.head_dim), dtype=np.float32), config.text_config.dtype),
    }
    if use_attention_mask:
        inputs["attention_mask"] = _float_input(np.zeros((settings.batch * config.text_config.num_attention_heads, 1, total_len), dtype=np.float32), config.text_config.dtype)
    if variant == "static":
        inputs["cache_seqlens"] = np.full((settings.batch,), settings.past_len, dtype=np.int32)
        cache_buffer_len = total_len
    else:
        cache_buffer_len = settings.past_len
    for layer_idx in range(config.text_config.num_hidden_layers):
        shape = (settings.batch, config.text_config.num_key_value_heads, cache_buffer_len, config.text_config.head_dim)
        inputs[f"past_key_{layer_idx}"] = _float_input(rng.normal(0.0, 0.01, shape).astype(np.float32), config.text_config.dtype)
        inputs[f"past_value_{layer_idx}"] = _float_input(rng.normal(0.0, 0.01, shape).astype(np.float32), config.text_config.dtype)
    return inputs


def _profiling_shape_scenarios(settings: WorkflowSettings, config, *, cache_variant: str, use_attention_mask: bool) -> list[dict[str, object]]:
    batch_values = _equal_interval_buckets(settings.min_batch, settings.max_batch, count=settings.batch_bucket_count)
    past_values = _equal_interval_buckets(settings.min_past_len, settings.max_past_len, count=settings.past_bucket_count)
    scenarios = []
    for batch_size in batch_values:
        for past_len in past_values:
            total_len = int(past_len) + 1
            overrides = {
                "input_ids": [int(batch_size), 1],
                "cos": [int(batch_size), 1, int(config.text_config.head_dim)],
                "sin": [int(batch_size), 1, int(config.text_config.head_dim)],
            }
            if use_attention_mask:
                overrides["attention_mask"] = [int(batch_size) * int(config.text_config.num_attention_heads), 1, total_len]
            if cache_variant == "static":
                overrides["cache_seqlens"] = [int(batch_size)]
                buffer_len = total_len
            else:
                buffer_len = int(past_len)
            for layer_idx in range(int(config.text_config.num_hidden_layers)):
                overrides[f"past_key_{layer_idx}"] = [int(batch_size), int(config.text_config.num_key_value_heads), buffer_len, int(config.text_config.head_dim)]
                overrides[f"past_value_{layer_idx}"] = [int(batch_size), int(config.text_config.num_key_value_heads), buffer_len, int(config.text_config.head_dim)]
            scenarios.append(
                {
                    "source": "workflow_equal_interval_buckets",
                    "case_id": f"batch={int(batch_size)}_past_len={int(past_len)}",
                    "dim_values": {"batch": int(batch_size), "past_len": int(past_len), "total_len": total_len},
                    "overrides": overrides,
                }
            )
    return scenarios


def _normalize_cache_variant(cache_variant: str) -> str:
    normalized = str(cache_variant).strip().lower()
    if normalized not in _CACHE_VARIANTS:
        raise ValueError(f"cache_variant must be one of {sorted(_CACHE_VARIANTS)}, got {cache_variant!r}")
    return normalized


def _resolve_settings(
    *,
    snapshot: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    batch: int | None = None,
    min_batch: int = 1,
    max_batch: int | None = None,
    past_len: int | None = None,
    min_past_len: int = 1,
    max_past_len: int | None = None,
    dtype: str = "bfloat16",
    batch_bucket_count: int = 3,
    past_bucket_count: int = 3,
) -> WorkflowSettings:
    resolved_snapshot, resolved_config_path, resolved_checkpoint_path = resolve_snapshot_paths(
        snapshot=snapshot,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    resolved_min_batch = int(min_batch)
    resolved_batch = resolved_min_batch if batch is None else int(batch)
    resolved_max_batch = max(resolved_batch, 4) if max_batch is None else int(max_batch)
    resolved_min_past_len = int(min_past_len)
    resolved_past_len = resolved_min_past_len if past_len is None else int(past_len)
    resolved_max_past_len = max(resolved_past_len, 64) if max_past_len is None else int(max_past_len)
    return WorkflowSettings(
        snapshot=resolved_snapshot,
        config_path=resolved_config_path,
        checkpoint_path=resolved_checkpoint_path,
        batch=resolved_batch,
        min_batch=resolved_min_batch,
        max_batch=resolved_max_batch,
        past_len=resolved_past_len,
        min_past_len=resolved_min_past_len,
        max_past_len=resolved_max_past_len,
        dtype=str(dtype),
        batch_bucket_count=int(batch_bucket_count),
        past_bucket_count=int(past_bucket_count),
    )
