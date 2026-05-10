from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Sequence


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
ShapeSpecDim = int | dict[str, Any]
ShapeLike = Sequence[ShapeDim]
SymbolicInt = int | dict[str, Any]

_INT_EXPR_KIND = "int_expr"
_INT_EXPR_OPS = frozenset({"add", "sub", "mul", "div"})


@dataclass(frozen=True)
class Shape:
    """Canonical shape metadata used by frontend tensors and runtime helpers."""

    _shape_spec: tuple[ShapeSpecDim, ...]

    def __init__(self, dims: ShapeLike | "Shape"):
        if isinstance(dims, Shape):
            normalized = tuple(dims.to_json())
        else:
            normalized = tuple(normalize_dim(dim) for dim in dims)
        object.__setattr__(self, "_shape_spec", normalized)

    @property
    def rank(self) -> int:
        return len(self._shape_spec)

    @property
    def shape_spec(self) -> list[ShapeSpecDim]:
        return self.to_json()

    @property
    def max_shape(self) -> list[int]:
        return max_shape(self._shape_spec)

    @property
    def dynamic(self) -> bool:
        return is_dynamic_shape(self._shape_spec)

    @property
    def numel(self) -> int:
        return shape_numel(self.max_shape)

    @property
    def constraints(self) -> list[dict[str, Any]]:
        return shape_constraints(self._shape_spec)

    def validate_max_shape(self, expected: Sequence[int]) -> None:
        actual = self.max_shape
        if actual != list(expected):
            raise ValueError(f"shape_spec max shape {actual} does not match shape {list(expected)}")

    def validate_runtime(self, name: str, shape: Iterable[int]) -> tuple[int, ...]:
        return validate_runtime_shape(name, shape, {"name": name, "shape": self.max_shape, "shape_spec": self._shape_spec})

    def to_json(self) -> list[ShapeSpecDim]:
        result: list[ShapeSpecDim] = []
        for dim in self._shape_spec:
            result.append(dim if isinstance(dim, int) else dict(dim))
        return result

    def __len__(self) -> int:
        return len(self._shape_spec)

    def __iter__(self) -> Iterator[ShapeSpecDim]:
        return iter(self.to_json())

    def __getitem__(self, index: int | slice) -> ShapeSpecDim | list[ShapeSpecDim]:
        values = self.to_json()
        return values[index]


def normalize_shape(shape: ShapeLike | Shape) -> list[ShapeSpecDim]:
    if isinstance(shape, Shape):
        return shape.to_json()
    result: list[ShapeSpecDim] = []
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


def symbolic_int_expr(op: str, lhs: Any, rhs: Any) -> SymbolicInt:
    """Build a serializable symbolic integer expression.

    The ``div`` op uses Python floor-division semantics. This helper is only a
    frontend representation scaffold; expression dicts are intentionally not
    valid shape dimensions yet.
    """

    if op not in _INT_EXPR_OPS:
        raise ValueError(f"Unsupported symbolic integer op: {op}")
    left = normalize_symbolic_int(lhs, "lhs")
    right = normalize_symbolic_int(rhs, "rhs")
    if op == "div" and right == 0:
        raise ZeroDivisionError("symbolic integer div by zero")
    if isinstance(left, int) and isinstance(right, int):
        if op == "add":
            return left + right
        if op == "sub":
            return left - right
        if op == "mul":
            return left * right
        return left // right
    return {"kind": _INT_EXPR_KIND, "op": op, "lhs": left, "rhs": right}


def normalize_symbolic_int(value: Any, name: str = "symbolic integer") -> SymbolicInt:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer or symbolic dimension, got bool")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, Mapping):
        kind = value.get("kind")
        if kind == "dim":
            return normalize_dim(value)
        if kind == _INT_EXPR_KIND:
            op = value.get("op")
            if op not in _INT_EXPR_OPS:
                raise ValueError(f"Unsupported symbolic integer op: {op}")
            lhs = normalize_symbolic_int(value.get("lhs"), "lhs")
            rhs = normalize_symbolic_int(value.get("rhs"), "rhs")
            if op == "div" and rhs == 0:
                raise ZeroDivisionError("symbolic integer div by zero")
            return {
                "kind": _INT_EXPR_KIND,
                "op": op,
                "lhs": lhs,
                "rhs": rhs,
            }
        raise ValueError(f"Unsupported symbolic integer mapping: {value!r}")
    raise TypeError(f"{name} must be an integer or symbolic dimension, got {type(value).__name__}")


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


def runtime_shape_tuple(shape: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(dim) for dim in shape)
    for axis, dim in enumerate(values):
        if dim <= 0:
            raise ValueError(f"Runtime shape axis {axis} must be positive, got {dim}")
    return values


def validate_runtime_shape(name: str, shape: Iterable[int], spec: Mapping[str, Any]) -> tuple[int, ...]:
    actual_shape = runtime_shape_tuple(shape)
    shape_spec = spec.get("shape_spec", spec["shape"])
    if len(actual_shape) != len(shape_spec):
        raise ValueError(f"{name} rank mismatch: got {len(actual_shape)}, expected {len(shape_spec)}")
    for axis, (actual, dim_spec) in enumerate(zip(actual_shape, shape_spec)):
        if isinstance(dim_spec, int):
            if actual != int(dim_spec):
                raise ValueError(f"{name} axis {axis} has dim {actual}, expected static dim {dim_spec}")
            continue
        dim_name = str(dim_spec["name"])
        min_dim = int(dim_spec["min"])
        max_dim = int(dim_spec["max"])
        divisible_by = int(dim_spec.get("divisible_by", 1))
        if actual < min_dim or actual > max_dim:
            raise ValueError(f"{name} axis {axis} ({dim_name}) has dim {actual}, expected [{min_dim}, {max_dim}]")
        if actual % divisible_by != 0:
            raise ValueError(f"{name} axis {axis} ({dim_name}) has dim {actual}, expected divisible by {divisible_by}")
    return actual_shape


def infer_output_shape(
    output_spec: Mapping[str, Any],
    input_specs: Sequence[Mapping[str, Any]],
    input_shapes: Mapping[str, Iterable[int]],
) -> tuple[int, ...]:
    dim_values: dict[str, int] = {}
    for spec in input_specs:
        input_name = str(spec["name"])
        if input_name not in input_shapes:
            raise ValueError(f"Missing runtime shape for input {input_name}")
        actual_shape = validate_runtime_shape(input_name, input_shapes[input_name], spec)
        shape_spec = spec.get("shape_spec", spec["shape"])
        for axis, (actual, dim_spec) in enumerate(zip(actual_shape, shape_spec)):
            if isinstance(dim_spec, int):
                continue
            dim_name = str(dim_spec["name"])
            existing = dim_values.get(dim_name)
            if existing is not None and existing != int(actual):
                raise ValueError(
                    f"Dynamic dimension {dim_name} has inconsistent values {existing} and {actual} "
                    f"while reading input {input_name} axis {axis}"
                )
            dim_values[dim_name] = int(actual)
    result = []
    for dim_spec in output_spec.get("shape_spec", output_spec["shape"]):
        if isinstance(dim_spec, int):
            result.append(int(dim_spec))
        else:
            result.append(dim_values.get(str(dim_spec["name"]), int(dim_spec["max"])))
    return tuple(result)
