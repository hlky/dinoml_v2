from __future__ import annotations

from typing import Any, Mapping, Sequence


LAYOUT_SCHEMA_VERSION = 1


def contiguous_strides(shape: Sequence[int]) -> list[int]:
    strides = []
    stride = 1
    for dim in reversed([int(value) for value in shape]):
        strides.append(stride)
        stride *= dim
    return list(reversed(strides))


def dense_layout(shape: Sequence[int], *, alignment: int | None = None) -> dict[str, Any]:
    layout: dict[str, Any] = {
        "schema_version": LAYOUT_SCHEMA_VERSION,
        "kind": "dense",
        "order": "row_major",
        "strides": contiguous_strides(shape),
        "storage_offset": 0,
    }
    if alignment is not None:
        layout["alignment"] = int(alignment)
    return layout


def validate_layout(layout: Mapping[str, Any], shape: Sequence[int]) -> dict[str, Any]:
    if layout.get("schema_version") != LAYOUT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported layout schema: {layout.get('schema_version')}")
    if layout.get("kind") != "dense":
        raise ValueError(f"Unsupported layout kind: {layout.get('kind')}")
    if layout.get("order") != "row_major":
        raise ValueError(f"Unsupported layout order: {layout.get('order')}")
    if int(layout.get("storage_offset", 0)) != 0:
        raise ValueError("Dense layout storage_offset must be 0")
    expected_strides = contiguous_strides(shape)
    actual_strides = [int(stride) for stride in layout.get("strides", [])]
    if actual_strides != expected_strides:
        raise ValueError(f"Dense layout strides {actual_strides} do not match row-major strides {expected_strides}")
    normalized = dense_layout(shape)
    if "alignment" in layout:
        alignment = int(layout["alignment"])
        if alignment <= 0:
            raise ValueError("Dense layout alignment must be positive")
        normalized["alignment"] = alignment
    return normalized
