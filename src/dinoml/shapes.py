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
        kind = dim.get("kind")
        if kind == "dim":
            return _normalize_dim_mapping(dim)
        if kind == _INT_EXPR_KIND:
            expr = normalize_symbolic_int(dim)
            interval = symbolic_int_interval(expr)
            if interval[0] <= 0:
                raise ValueError(f"Symbolic shape expression minimum must be positive, got {interval[0]} for {expr!r}")
            return expr
        raise ValueError(f"Unsupported shape dimension mapping: {dim!r}")
    raise ValueError(f"Unsupported shape dimension type: {type(dim).__name__}")


def _normalize_dim_mapping(dim: Mapping[str, Any]) -> dict[str, Any]:
    return Dim(
        name=str(dim["name"]),
        min=int(dim["min"]),
        max=int(dim["max"]),
        divisible_by=int(dim.get("divisible_by", 1)),
        typical=None if dim.get("typical") is None else int(dim["typical"]),
        buckets=tuple(int(bucket) for bucket in dim.get("buckets", ())),
    ).to_json()


def symbolic_int_expr(op: str, lhs: Any, rhs: Any) -> SymbolicInt:
    """Build a serializable symbolic integer expression.

    The ``div`` op uses Python floor-division semantics. Dynamic expression
    dicts are JSON-compatible and may be used as bounded shape dimensions when
    their interval minimum is positive.
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


def symbolic_int_interval(value: SymbolicInt) -> tuple[int, int]:
    if isinstance(value, int):
        return (int(value), int(value))
    kind = value.get("kind")
    if kind == "dim":
        dim = _normalize_dim_mapping(value)
        return (int(dim["min"]), int(dim["max"]))
    if kind != _INT_EXPR_KIND:
        raise ValueError(f"Unsupported symbolic integer mapping: {value!r}")
    op = str(value["op"])
    lhs_min, lhs_max = symbolic_int_interval(value["lhs"])
    rhs_min, rhs_max = symbolic_int_interval(value["rhs"])
    if op == "add":
        return (lhs_min + rhs_min, lhs_max + rhs_max)
    if op == "sub":
        return (lhs_min - rhs_max, lhs_max - rhs_min)
    if op == "mul":
        candidates = (lhs_min * rhs_min, lhs_min * rhs_max, lhs_max * rhs_min, lhs_max * rhs_max)
        return (min(candidates), max(candidates))
    if op == "div":
        if rhs_min <= 0 <= rhs_max:
            raise ZeroDivisionError(f"symbolic integer div denominator interval contains zero: [{rhs_min}, {rhs_max}]")
        candidates = (lhs_min // rhs_min, lhs_min // rhs_max, lhs_max // rhs_min, lhs_max // rhs_max)
        return (min(candidates), max(candidates))
    raise ValueError(f"Unsupported symbolic integer op: {op}")


def evaluate_symbolic_int(value: SymbolicInt, dim_values: Mapping[str, int]) -> int:
    if isinstance(value, int):
        return int(value)
    kind = value.get("kind")
    if kind == "dim":
        dim_name = str(value["name"])
        if dim_name not in dim_values:
            raise ValueError(f"Missing runtime value for dynamic dimension {dim_name}")
        return int(dim_values[dim_name])
    if kind != _INT_EXPR_KIND:
        raise ValueError(f"Unsupported symbolic integer mapping: {value!r}")
    op = str(value["op"])
    lhs = evaluate_symbolic_int(value["lhs"], dim_values)
    rhs = evaluate_symbolic_int(value["rhs"], dim_values)
    if op == "add":
        return lhs + rhs
    if op == "sub":
        return lhs - rhs
    if op == "mul":
        return lhs * rhs
    if op == "div":
        if rhs == 0:
            raise ZeroDivisionError("symbolic integer div by zero")
        return lhs // rhs
    raise ValueError(f"Unsupported symbolic integer op: {op}")


def max_shape(shape_spec: Sequence[int | Mapping[str, Any]]) -> list[int]:
    result = []
    for dim in shape_spec:
        if isinstance(dim, int):
            result.append(dim)
        elif dim.get("kind") == _INT_EXPR_KIND:
            result.append(symbolic_int_interval(normalize_symbolic_int(dim))[1])
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
        if isinstance(dim, int):
            continue
        if dim.get("kind") == "dim":
            item = dict(dim)
            item["axis"] = index
            constraints.append(item)
        elif dim.get("kind") == _INT_EXPR_KIND:
            for leaf in _symbolic_dim_leaves(dim):
                item = dict(leaf)
                item["axis"] = index
                constraints.append(item)
    return constraints


def _symbolic_dim_leaves(value: SymbolicInt) -> list[dict[str, Any]]:
    if isinstance(value, int):
        return []
    kind = value.get("kind")
    if kind == "dim":
        return [_normalize_dim_mapping(value)]
    if kind == _INT_EXPR_KIND:
        return [*_symbolic_dim_leaves(value["lhs"]), *_symbolic_dim_leaves(value["rhs"])]
    raise ValueError(f"Unsupported symbolic integer mapping: {value!r}")


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
    dim_values: dict[str, int] = {}
    for axis, (actual, dim_spec) in enumerate(zip(actual_shape, shape_spec)):
        if isinstance(dim_spec, int):
            if actual != int(dim_spec):
                raise ValueError(f"{name} axis {axis} has dim {actual}, expected static dim {dim_spec}")
            continue
        if dim_spec.get("kind") == _INT_EXPR_KIND:
            expected = evaluate_symbolic_int(dim_spec, dim_values)
            if expected <= 0:
                raise ValueError(f"{name} axis {axis} symbolic expression evaluated to non-positive dim {expected}")
            if actual != expected:
                raise ValueError(f"{name} axis {axis} has dim {actual}, expected symbolic dim {expected}")
            continue
        dim_name = str(dim_spec["name"])
        min_dim = int(dim_spec["min"])
        max_dim = int(dim_spec["max"])
        divisible_by = int(dim_spec.get("divisible_by", 1))
        if actual < min_dim or actual > max_dim:
            raise ValueError(f"{name} axis {axis} ({dim_name}) has dim {actual}, expected [{min_dim}, {max_dim}]")
        if actual % divisible_by != 0:
            raise ValueError(f"{name} axis {axis} ({dim_name}) has dim {actual}, expected divisible by {divisible_by}")
        existing = dim_values.get(dim_name)
        if existing is not None and existing != int(actual):
            raise ValueError(f"{name} axis {axis} ({dim_name}) has dim {actual}, expected same value as earlier {existing}")
        dim_values[dim_name] = int(actual)
    return actual_shape


def validate_runtime_input_shapes(
    input_specs: Sequence[Mapping[str, Any]],
    input_shapes: Mapping[str, Iterable[int]],
) -> dict[str, tuple[int, ...]]:
    actual_shapes: dict[str, tuple[int, ...]] = {}
    dim_values: dict[str, int] = {}
    pending_exprs: list[tuple[str, int, int, Mapping[str, Any]]] = []
    for spec in input_specs:
        input_name = str(spec["name"])
        if input_name not in input_shapes:
            raise ValueError(f"Missing runtime shape for input {input_name}")
        actual_shape = runtime_shape_tuple(input_shapes[input_name])
        shape_spec = spec.get("shape_spec", spec["shape"])
        if len(actual_shape) != len(shape_spec):
            raise ValueError(f"{input_name} rank mismatch: got {len(actual_shape)}, expected {len(shape_spec)}")
        actual_shapes[input_name] = actual_shape
        for axis, (actual, dim_spec) in enumerate(zip(actual_shape, shape_spec)):
            if isinstance(dim_spec, int):
                if actual != int(dim_spec):
                    raise ValueError(f"{input_name} axis {axis} has dim {actual}, expected static dim {dim_spec}")
                continue
            if dim_spec.get("kind") == _INT_EXPR_KIND:
                pending_exprs.append((input_name, axis, int(actual), dim_spec))
                continue
            dim_name = str(dim_spec["name"])
            min_dim = int(dim_spec["min"])
            max_dim = int(dim_spec["max"])
            divisible_by = int(dim_spec.get("divisible_by", 1))
            if actual < min_dim or actual > max_dim:
                raise ValueError(f"{input_name} axis {axis} ({dim_name}) has dim {actual}, expected [{min_dim}, {max_dim}]")
            if actual % divisible_by != 0:
                raise ValueError(f"{input_name} axis {axis} ({dim_name}) has dim {actual}, expected divisible by {divisible_by}")
            existing = dim_values.get(dim_name)
            if existing is not None and existing != int(actual):
                raise ValueError(
                    f"Dynamic dimension {dim_name} has inconsistent values {existing} and {actual} "
                    f"while reading input {input_name} axis {axis}"
                )
            dim_values[dim_name] = int(actual)
    for input_name, axis, actual, dim_spec in pending_exprs:
        expected = evaluate_symbolic_int(dim_spec, dim_values)
        if expected <= 0:
            raise ValueError(f"{input_name} axis {axis} symbolic expression evaluated to non-positive dim {expected}")
        if actual != expected:
            raise ValueError(f"{input_name} axis {axis} has dim {actual}, expected symbolic dim {expected}")
    return actual_shapes


def infer_output_shape(
    output_spec: Mapping[str, Any],
    input_specs: Sequence[Mapping[str, Any]],
    input_shapes: Mapping[str, Iterable[int]],
) -> tuple[int, ...]:
    dim_values: dict[str, int] = {}
    actual_shapes = validate_runtime_input_shapes(input_specs, input_shapes)
    for spec in input_specs:
        input_name = str(spec["name"])
        actual_shape = actual_shapes[input_name]
        shape_spec = spec.get("shape_spec", spec["shape"])
        for axis, (actual, dim_spec) in enumerate(zip(actual_shape, shape_spec)):
            if isinstance(dim_spec, int):
                continue
            if dim_spec.get("kind") == _INT_EXPR_KIND:
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
        elif dim_spec.get("kind") == _INT_EXPR_KIND:
            value = evaluate_symbolic_int(dim_spec, dim_values)
            if value <= 0:
                raise ValueError(f"Output symbolic shape expression evaluated to non-positive dim {value}")
            result.append(value)
        else:
            dim_name = str(dim_spec["name"])
            result.append(dim_values.get(dim_name, int(dim_spec["max"])))
    return tuple(result)
