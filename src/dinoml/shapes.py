from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Dim:
    name: str
    min: int
    max: int
    divisible_by: int = 1
    typical: int | None = None
    buckets: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Dim.name must not be empty")
        if self.min <= 0 or self.max <= 0:
            raise ValueError(f"Dim {self.name} bounds must be positive")
        if self.min > self.max:
            raise ValueError(f"Dim {self.name} min must be <= max")
        if self.divisible_by <= 0:
            raise ValueError(f"Dim {self.name} divisible_by must be positive")
        if self.min % self.divisible_by != 0 or self.max % self.divisible_by != 0:
            raise ValueError(f"Dim {self.name} bounds must be divisible by {self.divisible_by}")
        if self.typical is not None and not (self.min <= self.typical <= self.max):
            raise ValueError(f"Dim {self.name} typical value must be within [min, max]")
        for bucket in self.buckets:
            if not (self.min <= bucket <= self.max):
                raise ValueError(f"Dim {self.name} bucket {bucket} is outside [min, max]")
            if bucket % self.divisible_by != 0:
                raise ValueError(f"Dim {self.name} bucket {bucket} is not divisible by {self.divisible_by}")

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": "dim",
            "name": self.name,
            "min": self.min,
            "max": self.max,
            "divisible_by": self.divisible_by,
        }
        if self.typical is not None:
            data["typical"] = self.typical
        if self.buckets:
            data["buckets"] = list(self.buckets)
        return data


ShapeDim = int | Dim | Mapping[str, Any]
Shape = Sequence[ShapeDim]


def normalize_shape(shape: Shape) -> list[int | dict[str, Any]]:
    result: list[int | dict[str, Any]] = []
    for dim in shape:
        result.append(normalize_dim(dim))
    if not result:
        return result
    return result


def normalize_dim(dim: ShapeDim) -> int | dict[str, Any]:
    if isinstance(dim, bool):
        raise ValueError(f"Invalid boolean shape dimension: {dim!r}")
    if isinstance(dim, int):
        if dim <= 0:
            raise ValueError(f"Static shape dimensions must be positive, got {dim}")
        return dim
    if isinstance(dim, Dim):
        return dim.to_json()
    if isinstance(dim, Mapping):
        if dim.get("kind") != "dim":
            raise ValueError(f"Unsupported shape dimension mapping: {dim!r}")
        return Dim(
            name=str(dim["name"]),
            min=int(dim["min"]),
            max=int(dim["max"]),
            divisible_by=int(dim.get("divisible_by", 1)),
            typical=None if dim.get("typical") is None else int(dim["typical"]),
            buckets=tuple(int(bucket) for bucket in dim.get("buckets", ())),
        ).to_json()
    raise ValueError(f"Unsupported shape dimension type: {type(dim).__name__}")


def max_shape(shape_spec: Sequence[int | Mapping[str, Any]]) -> list[int]:
    result = []
    for dim in shape_spec:
        if isinstance(dim, int):
            result.append(dim)
        else:
            result.append(int(dim["max"]))
    return result


def is_dynamic_shape(shape_spec: Sequence[int | Mapping[str, Any]]) -> bool:
    return any(not isinstance(dim, int) for dim in shape_spec)


def validate_shape_spec(shape_spec: Sequence[int | Mapping[str, Any]], expected_max_shape: Sequence[int] | None = None) -> None:
    normalized = normalize_shape(shape_spec)
    if expected_max_shape is not None and max_shape(normalized) != list(expected_max_shape):
        raise ValueError(f"shape_spec max shape {max_shape(normalized)} does not match shape {list(expected_max_shape)}")


def shape_constraints(shape_spec: Sequence[int | Mapping[str, Any]]) -> list[dict[str, Any]]:
    constraints = []
    for index, dim in enumerate(shape_spec):
        if not isinstance(dim, int):
            item = dict(dim)
            item["axis"] = index
            constraints.append(item)
    return constraints


def shape_numel(shape: Iterable[int]) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel
