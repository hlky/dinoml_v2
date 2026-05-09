from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dinoml.ir import array_to_storage, bfloat16_storage_to_float32, normalize_dtype


@dataclass(frozen=True)
class MaterializedConstant:
    array: np.ndarray
    storage: dict[str, Any] | None = None


@dataclass(frozen=True)
class GGUFConstant:
    path: str | Path
    tensor: str
    logical_dtype: str | None = None
    shape: Sequence[int] | None = None
    qtype: str | None = None
    encoded_nbytes: int | None = None
    n_per_row: int | None = None
    materialization: str = "dequantize_full_before_launch"
    residency: str = "eager_dense_device"

    def materialize(self, dtype: str, shape: Sequence[int]) -> MaterializedConstant:
        logical_dtype = normalize_dtype(dtype)
        expected_shape = tuple(int(dim) for dim in shape)
        if self.logical_dtype is not None and normalize_dtype(self.logical_dtype) != logical_dtype:
            raise ValueError(
                f"GGUF constant {self.tensor!r} has logical dtype {normalize_dtype(self.logical_dtype)}, "
                f"expected {logical_dtype}"
            )
        if self.shape is not None:
            source_shape = tuple(int(dim) for dim in self.shape)
            if source_shape != expected_shape:
                raise ValueError(f"GGUF constant {self.tensor!r} has shape {source_shape}, expected {expected_shape}")
        array, observed = _materialize_gguf_constant(self, logical_dtype, expected_shape)
        storage_array = array_to_storage(array, logical_dtype)
        if storage_array.shape != expected_shape:
            raise ValueError(
                f"GGUF constant {self.tensor!r} materialized shape {storage_array.shape}, "
                f"expected {expected_shape}"
            )
        metadata = self._metadata(logical_dtype, expected_shape, observed)
        return MaterializedConstant(storage_array, metadata)

    def _metadata(
        self,
        dtype: str,
        shape: tuple[int, ...],
        observed: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._validate_observed_hints(observed)
        metadata: dict[str, Any] = {
            "kind": "gguf",
            "path": str(self.path),
            "tensor": self.tensor,
            "logical_dtype": dtype,
            "shape": list(shape),
            "materialization": self.materialization,
            "residency": self.residency,
        }
        optional = {
            "qtype": observed.get("qtype"),
            "qtype_value": observed.get("qtype_value"),
            "encoded_nbytes": observed.get("encoded_nbytes"),
            "gguf_shape": observed.get("gguf_shape"),
            "data_offset": observed.get("data_offset"),
            "n_per_row": observed.get("n_per_row"),
        }
        for key, value in optional.items():
            if value is not None:
                metadata[key] = value
        return metadata

    def _validate_observed_hints(self, observed: Mapping[str, Any]) -> None:
        if self.qtype is not None and self.qtype != observed.get("qtype"):
            raise ValueError(
                f"GGUF constant {self.tensor!r} expected qtype {self.qtype}, "
                f"observed {observed.get('qtype')}"
            )
        if self.encoded_nbytes is not None and self.encoded_nbytes != observed.get("encoded_nbytes"):
            raise ValueError(
                f"GGUF constant {self.tensor!r} expected {self.encoded_nbytes} encoded bytes, "
                f"observed {observed.get('encoded_nbytes')}"
            )
        if self.n_per_row is not None and self.n_per_row != observed.get("n_per_row"):
            raise ValueError(
                f"GGUF constant {self.tensor!r} expected n_per_row {self.n_per_row}, "
                f"observed {observed.get('n_per_row')}"
            )


def gguf_constant(
    path: str | Path,
    tensor: str,
    *,
    logical_dtype: str | None = None,
    shape: Sequence[int] | None = None,
    qtype: str | None = None,
    encoded_nbytes: int | None = None,
    n_per_row: int | None = None,
    materialization: str = "dequantize_full_before_launch",
    residency: str = "eager_dense_device",
) -> GGUFConstant:
    return GGUFConstant(
        path=path,
        tensor=tensor,
        logical_dtype=logical_dtype,
        shape=shape,
        qtype=qtype,
        encoded_nbytes=encoded_nbytes,
        n_per_row=n_per_row,
        materialization=materialization,
        residency=residency,
    )


def materialize_constant_value(value: Any, dtype: str, shape: Sequence[int]) -> MaterializedConstant:
    if isinstance(value, GGUFConstant):
        return value.materialize(dtype, shape)
    materialize = getattr(value, "materialize", None)
    if callable(materialize):
        result = materialize(dtype, shape)
        if isinstance(result, MaterializedConstant):
            return result
        return MaterializedConstant(array_to_storage(result, dtype), getattr(value, "storage", None))
    return MaterializedConstant(array_to_storage(value, dtype), None)


def constant_source_from_storage(storage: Mapping[str, Any]) -> GGUFConstant | None:
    if storage.get("kind") != "gguf":
        return None
    source = storage.get("source", {})
    if not isinstance(source, Mapping):
        source = {}
    path = storage.get("path") or source.get("path")
    tensor = storage.get("tensor") or source.get("tensor")
    if path is None or tensor is None:
        raise ValueError("GGUF storage metadata requires path and tensor")
    return GGUFConstant(
        path=path,
        tensor=str(tensor),
        logical_dtype=_optional_str(storage.get("logical_dtype")),
        shape=_optional_shape(storage.get("shape")),
        qtype=_optional_str(storage.get("qtype")),
        encoded_nbytes=_optional_int(storage.get("encoded_nbytes")),
        n_per_row=_optional_int(storage.get("n_per_row")),
        materialization=str(storage.get("materialization", "dequantize_full_before_launch")),
        residency=str(storage.get("residency", "eager_dense_device")),
    )


def _materialize_gguf_constant(
    source: GGUFConstant,
    dtype: str,
    expected_shape: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import libgguf  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "GGUF constants require the libgguf Python package. Install hlky/libgguf "
            "or bind a dense NumPy value before compiling."
        ) from exc

    gguf = libgguf.open_gguf(source.path)
    tensor_info = gguf.get_tensor(source.tensor)
    if tensor_info is None:
        raise ValueError(f"GGUF tensor not found: {source.tensor!r}")
    gguf_shape = tuple(int(dim) for dim in getattr(tensor_info, "shape"))
    logical_shape = _gguf_logical_shape(gguf_shape)
    if logical_shape != expected_shape:
        raise ValueError(
            f"GGUF tensor {source.tensor!r} has logical shape {logical_shape}, expected {expected_shape} "
            f"(stored GGUF shape {gguf_shape})"
        )
    raw = gguf.read_tensor_bytes(tensor_info)
    qtype = int(getattr(tensor_info, "qtype_value"))
    qtype_name = str(getattr(tensor_info, "qtype"))
    n_per_row = int(gguf_shape[0]) if gguf_shape else 1
    n_rows = int(np.prod(expected_shape[:-1], dtype=np.int64)) if len(expected_shape) > 1 else 1
    encoded = np.frombuffer(raw, dtype=np.uint8)
    decoded = _decode_gguf_rows(libgguf, encoded, qtype, qtype_name, n_rows, n_per_row, expected_shape)
    return decoded, {
        "qtype": qtype_name,
        "qtype_value": qtype,
        "encoded_nbytes": len(raw),
        "gguf_shape": list(gguf_shape),
        "data_offset": int(getattr(tensor_info, "data_offset")),
        "n_per_row": n_per_row,
    }


def _gguf_logical_shape(gguf_shape: Sequence[int]) -> tuple[int, ...]:
    if not gguf_shape:
        return ()
    return tuple(reversed(tuple(int(dim) for dim in gguf_shape)))


def _decode_gguf_rows(
    libgguf: Any,
    encoded: np.ndarray,
    qtype: int,
    qtype_name: str,
    n_rows: int,
    n_per_row: int,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    if qtype_name == "F32":
        return np.frombuffer(encoded, dtype=np.float32).reshape(expected_shape)
    if qtype_name == "F16":
        return np.frombuffer(encoded, dtype=np.float16).reshape(expected_shape)
    if qtype_name == "BF16":
        storage = np.frombuffer(encoded, dtype=np.uint16).reshape(expected_shape)
        return bfloat16_storage_to_float32(storage)
    bytes_per_row = int(libgguf.row_size(qtype, n_per_row))
    if bytes_per_row <= 0:
        raise ValueError(f"Unsupported GGUF row width {n_per_row} for qtype {qtype_name}")
    expected_nbytes = n_rows * bytes_per_row
    if encoded.nbytes != expected_nbytes:
        raise ValueError(
            f"GGUF tensor encoded byte length {encoded.nbytes} does not match expected "
            f"{expected_nbytes} for shape {expected_shape}"
        )
    rows = encoded.reshape((n_rows, bytes_per_row))
    return libgguf.dequantize_rows(rows, qtype, n_per_row=n_per_row).reshape(expected_shape)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_shape(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, SequenceABC) or isinstance(value, (str, bytes)):
        raise ValueError(f"GGUF storage shape must be a sequence of integers, got {value!r}")
    return tuple(int(dim) for dim in value)


__all__ = [
    "GGUFConstant",
    "MaterializedConstant",
    "constant_source_from_storage",
    "gguf_constant",
    "materialize_constant_value",
]
