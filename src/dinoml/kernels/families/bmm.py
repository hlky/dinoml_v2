from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from dinoml.ir import normalize_dtype


BMM_SUPPORTED_DTYPES = ("float16", "float32", "bfloat16")
BMM_LAYOUTS = ("ccc", "ccr", "crc", "crr", "rcc", "rcr", "rrc", "rrr")


@dataclass(frozen=True)
class BmmOpSpec:
    name: str
    base_layout: str
    layouts: Mapping[str, str]

    @property
    def input_count(self) -> int:
        return 2

    @property
    def a_layout(self) -> str:
        return self.base_layout[0]

    @property
    def b_layout(self) -> str:
        return self.base_layout[1]

    @property
    def c_layout(self) -> str:
        return self.base_layout[2]

    def validate_shapes(self, shapes: Sequence[Sequence[int]]) -> list[int]:
        if len(shapes) != self.input_count:
            raise ValueError(f"{self.name} expects exactly {self.input_count} inputs")
        a_shape, b_shape = shapes
        if len(a_shape) != 3 or len(b_shape) != 3:
            raise ValueError(f"{self.name} expects rank-3 A and B tensors")
        if any(int(dim) <= 0 for shape in (a_shape, b_shape) for dim in shape):
            raise ValueError(f"{self.name} dimensions must be positive")
        batch = _validate_batch(self.name, a_shape, b_shape)
        m = int(a_shape[_a_m_axis(self.a_layout)])
        k_a = int(a_shape[_a_k_axis(self.a_layout)])
        n = int(b_shape[_b_n_axis(self.b_layout)])
        k_b = int(b_shape[_b_k_axis(self.b_layout)])
        if k_a != k_b:
            raise ValueError(
                f"{self.name} expected compatible K dimensions, got A K={k_a} from {list(a_shape)} "
                f"and B K={k_b} from {list(b_shape)}"
            )
        return [batch, n, m] if self.c_layout == "c" else [batch, m, n]

    def output_shape_spec(self, shape_specs: Sequence[Sequence[Any]]) -> list[Any]:
        a_shape, b_shape = shape_specs
        batch = a_shape[0] if _dim_is_not_one(a_shape[0]) else b_shape[0]
        m = a_shape[_a_m_axis(self.a_layout)]
        n = b_shape[_b_n_axis(self.b_layout)]
        return [batch, n, m] if self.c_layout == "c" else [batch, m, n]

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_layout": self.base_layout,
            "layouts": dict(self.layouts),
            "input_count": self.input_count,
        }


def _bmm_op_spec(layout: str) -> BmmOpSpec:
    return BmmOpSpec(
        name=f"bmm_{layout}",
        base_layout=layout,
        layouts={
            "a": _layout_name(layout[0]),
            "b": _layout_name(layout[1]),
            "c": _layout_name(layout[2]),
        },
    )


def _a_m_axis(layout: str) -> int:
    return 2 if layout == "c" else 1


def _a_k_axis(layout: str) -> int:
    return 1 if layout == "c" else 2


def _b_n_axis(layout: str) -> int:
    return 1 if layout == "c" else 2


def _b_k_axis(layout: str) -> int:
    return 2 if layout == "c" else 1


def _layout_name(layout: str) -> str:
    if layout == "c":
        return "column"
    if layout == "r":
        return "row"
    raise ValueError(f"Unsupported BMM layout marker: {layout}")


def _validate_batch(op_name: str, a_shape: Sequence[int], b_shape: Sequence[int]) -> int:
    a_batch = int(a_shape[0])
    b_batch = int(b_shape[0])
    if a_batch == b_batch:
        return a_batch
    if a_batch == 1:
        return b_batch
    if b_batch == 1:
        return a_batch
    raise ValueError(
        f"{op_name} expected matching or broadcastable batch dimensions, got A batch={a_batch} and B batch={b_batch}"
    )


def _dim_is_not_one(dim: Any) -> bool:
    if isinstance(dim, int):
        return int(dim) != 1
    if isinstance(dim, Mapping):
        return int(dim["max"]) != 1
    return True


BMM_OP_SPECS: dict[str, BmmOpSpec] = {f"bmm_{layout}": _bmm_op_spec(layout) for layout in BMM_LAYOUTS}
BMM_OPS = tuple(BMM_OP_SPECS)


def bmm_op_spec(op_name: str) -> BmmOpSpec:
    try:
        return BMM_OP_SPECS[op_name]
    except KeyError as exc:
        supported = ", ".join(BMM_OPS)
        raise ValueError(f"Unsupported BMM op {op_name!r}; supported ops: {supported}") from exc


def normalize_bmm_dtype(dtype: str) -> str:
    normalized = normalize_dtype(dtype)
    if normalized not in BMM_SUPPORTED_DTYPES:
        supported = ", ".join(BMM_SUPPORTED_DTYPES)
        raise ValueError(f"Unsupported BMM dtype {dtype!r}; supported dtypes: {supported}")
    return normalized
