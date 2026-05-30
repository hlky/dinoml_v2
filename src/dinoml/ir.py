from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


IR_SCHEMA_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 1
RUNTIME_ABI_VERSION = 7
VIEW_METADATA_VERSION = 1
OUTPUT_SHAPE_REPORT_METADATA_VERSION = 1
VIEW_ONLY_TRANSFORMS = frozenset({"identity", "reshape", "flatten", "squeeze", "unsqueeze", "dynamic_slice"})


@dataclass(frozen=True)
class DTypeInfo:
    name: str
    nbytes: int
    runtime_enum: int
    numpy_dtype: str | None
    aliases: tuple[str, ...] = ()


DTYPES: tuple[DTypeInfo, ...] = (
    DTypeInfo("float16", 2, 1, "float16", ("half", "fp16")),
    DTypeInfo("float32", 4, 2, "float32", ("float", "fp32")),
    DTypeInfo("int32", 4, 3, "int32", ("int",)),
    DTypeInfo("int64", 8, 4, "int64", ("long",)),
    DTypeInfo("bool", 1, 5, "bool", ()),
    DTypeInfo("bfloat16", 2, 6, "uint16", ("bf16",)),
    DTypeInfo("float8_e4m3", 1, 7, None, ("float8_e4m3fn",)),
    DTypeInfo("float8_e5m2", 1, 8, None, ()),
)

_DTYPE_BY_NAME = {info.name: info for info in DTYPES}
_DTYPE_ALIASES = {alias: info.name for info in DTYPES for alias in info.aliases}
SUPPORTED_DTYPES = frozenset(_DTYPE_BY_NAME)


def dtype_nbytes(dtype: str) -> int:
    return _DTYPE_BY_NAME[normalize_dtype(dtype)].nbytes


def normalize_dtype(dtype: str) -> str:
    dtype = _DTYPE_ALIASES.get(dtype, dtype)
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return dtype


def dtype_runtime_enum(dtype: str) -> int:
    return _DTYPE_BY_NAME[normalize_dtype(dtype)].runtime_enum


def dtype_numpy(dtype: str) -> np.dtype:
    info = _DTYPE_BY_NAME[normalize_dtype(dtype)]
    if info.numpy_dtype is None:
        raise ValueError(f"Dtype {info.name} does not have a NumPy storage dtype in this MVP")
    return np.dtype(info.numpy_dtype)


def float32_to_bfloat16_storage(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    bits = array.view(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded >> np.uint32(16)).astype(np.uint16, copy=False)


def bfloat16_storage_to_float32(value: Any) -> np.ndarray:
    storage = np.asarray(value, dtype=np.uint16)
    bits = storage.astype(np.uint32, copy=False) << np.uint32(16)
    return bits.view(np.float32)


def array_to_storage(value: Any, dtype: str) -> np.ndarray:
    dtype = normalize_dtype(dtype)
    if dtype == "bfloat16":
        array = np.asarray(value)
        storage = array.astype(np.uint16, copy=False) if array.dtype == np.uint16 else float32_to_bfloat16_storage(array)
    else:
        storage = np.asarray(value, dtype=dtype_numpy(dtype))
    if not storage.flags.c_contiguous:
        storage = np.ascontiguousarray(storage)
    return storage


def array_from_storage(value: Any, dtype: str) -> np.ndarray:
    dtype = normalize_dtype(dtype)
    if dtype == "bfloat16":
        array = bfloat16_storage_to_float32(value)
    else:
        array = np.asarray(value, dtype=dtype_numpy(dtype))
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return array


def canonical_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"


def graph_hash(data: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def clone_ir(ir: Mapping[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(dict(ir))


@dataclass
class ModelSpec:
    name: str
    ir: Dict[str, Any]
    constants: Dict[str, Any]

    def clone(self) -> "ModelSpec":
        return ModelSpec(
            name=self.name,
            ir=clone_ir(self.ir),
            constants={name: _clone_constant_value(value) for name, value in self.constants.items()},
        )

    def canonical_json(self) -> str:
        return canonical_json(self.ir)

    def bind_constants(self, constants: Mapping[str, Any]) -> "ModelSpec":
        bound = {name: _clone_constant_value(value) for name, value in self.constants.items()}
        ir = clone_ir(self.ir)
        constant_specs = {constant["name"]: constant for constant in ir.get("constants", [])}
        for name, value in constants.items():
            if name not in constant_specs:
                raise ValueError(f"Unknown constant: {name}")
            spec = constant_specs[name]
            from dinoml.constant_sources import (
                GGUFConstant,
                GGUF_MATERIALIZATION_DEQUANTIZE_ON_GPU_BEFORE_LAUNCH,
                GGUF_RESIDENCY_MANUAL_RUNTIME_LOAD,
                materialize_gguf_encoded_constant,
                materialize_constant_value,
            )

            if (
                isinstance(value, GGUFConstant)
                and value.materialization == GGUF_MATERIALIZATION_DEQUANTIZE_ON_GPU_BEFORE_LAUNCH
                and value.residency == GGUF_RESIDENCY_MANUAL_RUNTIME_LOAD
            ):
                materialized = materialize_gguf_encoded_constant(value, spec["dtype"], spec["shape"])
                bound[name] = value
                if materialized.storage is not None:
                    spec["storage"] = materialized.storage
                continue

            materialized = materialize_constant_value(value, spec["dtype"], spec["shape"])
            array = materialized.array
            expected_shape = tuple(int(dim) for dim in spec["shape"])
            if array.shape != expected_shape:
                raise ValueError(f"Constant {name} has shape {array.shape}, expected {expected_shape}")
            bound[name] = array
            if materialized.storage is not None:
                spec["storage"] = materialized.storage
            else:
                spec.pop("storage", None)
        return ModelSpec(name=self.name, ir=ir, constants=bound)


def _clone_constant_value(value: Any) -> Any:
    if callable(getattr(value, "materialize", None)):
        return value
    return np.array(value, copy=True)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(canonical_json(data), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
