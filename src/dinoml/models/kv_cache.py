from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

import dinoml as dml
from dinoml.ir import array_to_storage, normalize_dtype


_KV_CACHE_DTYPES = frozenset({"float16", "float32", "bfloat16"})


@dataclass(frozen=True)
class StaticKvCacheSpec:
    num_layers: int
    batch: int
    num_key_value_heads: int
    max_cache_len: int
    head_dim: int
    dtype: str
    past_key_prefix: str = "past_key"
    past_value_prefix: str = "past_value"
    present_key_prefix: str = "present_key"
    present_value_prefix: str = "present_value"
    new_key_prefix: str = "new_key"
    new_value_prefix: str = "new_value"

    def __post_init__(self) -> None:
        for name in ("num_layers", "batch", "num_key_value_heads", "max_cache_len", "head_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        dtype = normalize_dtype(self.dtype)
        if dtype not in _KV_CACHE_DTYPES:
            raise ValueError(f"KV cache dtype must be one of {sorted(_KV_CACHE_DTYPES)}, got {self.dtype!r}")
        object.__setattr__(self, "dtype", dtype)

    def shape(self, cache_len: int | None = None) -> list[int]:
        return [
            int(self.batch),
            int(self.num_key_value_heads),
            int(self.max_cache_len if cache_len is None else cache_len),
            int(self.head_dim),
        ]

    def past_key_name(self, layer_idx: int) -> str:
        return f"{self.past_key_prefix}_{int(layer_idx)}"

    def past_value_name(self, layer_idx: int) -> str:
        return f"{self.past_value_prefix}_{int(layer_idx)}"

    def present_key_name(self, layer_idx: int) -> str:
        return f"{self.present_key_prefix}_{int(layer_idx)}"

    def present_value_name(self, layer_idx: int) -> str:
        return f"{self.present_value_prefix}_{int(layer_idx)}"

    def new_key_name(self, layer_idx: int) -> str:
        return f"{self.new_key_prefix}_{int(layer_idx)}"

    def new_value_name(self, layer_idx: int) -> str:
        return f"{self.new_value_prefix}_{int(layer_idx)}"


def static_kv_cache_input_specs(spec: StaticKvCacheSpec) -> dict[str, dml.TensorSpec]:
    return {
        name: dml.TensorSpec(spec.shape(), spec.dtype)
        for layer_idx in range(spec.num_layers)
        for name in (spec.past_key_name(layer_idx), spec.past_value_name(layer_idx))
    }


def empty_static_kv_cache(spec: StaticKvCacheSpec) -> dict[str, np.ndarray]:
    dtype = np.uint16 if spec.dtype == "bfloat16" else np.dtype(spec.dtype)
    return {
        name: np.zeros(spec.shape(), dtype=dtype)
        for layer_idx in range(spec.num_layers)
        for name in (spec.past_key_name(layer_idx), spec.past_value_name(layer_idx))
    }


def seed_static_kv_cache(
    outputs: Mapping[str, np.ndarray],
    spec: StaticKvCacheSpec,
    *,
    cache_len: int | None = None,
) -> dict[str, np.ndarray]:
    cache = empty_static_kv_cache(spec)
    for layer_idx in range(spec.num_layers):
        key = _cache_storage(np.asarray(outputs[spec.present_key_name(layer_idx)]), spec.dtype)
        value = _cache_storage(np.asarray(outputs[spec.present_value_name(layer_idx)]), spec.dtype)
        length = int(key.shape[2] if cache_len is None else cache_len)
        _validate_seed_shapes(spec, layer_idx, key, value, length)
        cache[spec.past_key_name(layer_idx)][:, :, :length, :] = key[:, :, :length, :]
        cache[spec.past_value_name(layer_idx)][:, :, :length, :] = value[:, :, :length, :]
    return cache


def write_static_kv_cache_update(
    cache: dict[str, np.ndarray],
    outputs: Mapping[str, np.ndarray],
    spec: StaticKvCacheSpec,
    *,
    position: int,
) -> None:
    slot = int(position)
    if slot < 0 or slot >= spec.max_cache_len:
        raise ValueError(f"KV cache update position {slot} is outside [0, {spec.max_cache_len})")
    for layer_idx in range(spec.num_layers):
        key = _cache_storage(np.asarray(outputs[spec.new_key_name(layer_idx)]), spec.dtype)
        value = _cache_storage(np.asarray(outputs[spec.new_value_name(layer_idx)]), spec.dtype)
        _validate_update_shapes(spec, layer_idx, key, value)
        cache[spec.past_key_name(layer_idx)][:, :, slot : slot + 1, :] = key
        cache[spec.past_value_name(layer_idx)][:, :, slot : slot + 1, :] = value


def append_static_kv_cache(past_key: Any, past_value: Any, new_key: Any, new_value: Any) -> tuple[Any, Any]:
    return dml.ops.concatenate([past_key, new_key], dim=2), dml.ops.concatenate([past_value, new_value], dim=2)


def _cache_storage(values: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bfloat16":
        if values.dtype == np.uint16:
            return np.ascontiguousarray(values)
        return array_to_storage(values.astype(np.float32, copy=False), "bfloat16")
    return values.astype(dtype, copy=False)


def _validate_seed_shapes(
    spec: StaticKvCacheSpec,
    layer_idx: int,
    key: np.ndarray,
    value: np.ndarray,
    cache_len: int,
) -> None:
    if cache_len < 0 or cache_len > spec.max_cache_len:
        raise ValueError(f"KV cache seed length {cache_len} is outside [0, {spec.max_cache_len}]")
    expected_prefix = tuple(spec.shape(cache_len=cache_len))
    for label, array in (("key", key), ("value", value)):
        if array.ndim != 4:
            raise ValueError(f"Layer {layer_idx} KV cache {label} must be rank 4, got shape {array.shape}")
        if tuple(array.shape[:2]) != expected_prefix[:2] or int(array.shape[3]) != expected_prefix[3]:
            raise ValueError(
                f"Layer {layer_idx} KV cache {label} shape {array.shape} does not match expected "
                f"[{spec.batch}, {spec.num_key_value_heads}, *, {spec.head_dim}]"
            )
        if int(array.shape[2]) < cache_len:
            raise ValueError(f"Layer {layer_idx} KV cache {label} length {array.shape[2]} is shorter than {cache_len}")


def _validate_update_shapes(spec: StaticKvCacheSpec, layer_idx: int, key: np.ndarray, value: np.ndarray) -> None:
    expected = tuple(spec.shape(cache_len=1))
    for label, array in (("key", key), ("value", value)):
        if tuple(array.shape) != expected:
            raise ValueError(f"Layer {layer_idx} KV cache update {label} has shape {array.shape}, expected {expected}")
