from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from dinoml.ir import (
    IR_SCHEMA_VERSION,
    ModelSpec,
    OUTPUT_SHAPE_REPORT_METADATA_VERSION,
    VIEW_METADATA_VERSION,
    array_to_storage,
    dtype_nbytes,
    normalize_dtype,
)
from dinoml.layout import dense_layout
from dinoml.shapes import Dim, Shape, shape_constraints, shape_numel


_CURRENT_BUILDER: Optional["GraphBuilder"] = None


@dataclass(frozen=True)
class TensorSpec:
    shape: Sequence[int | Dim | Mapping[str, Any]] | Shape
    dtype: str = "float32"
    shape_spec: list[int | dict[str, Any]] = field(init=False)
    max_shape: list[int] = field(init=False)
    dynamic: bool = field(init=False)
    _shape: Shape = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.shape:
            raise ValueError("TensorSpec.shape must not be empty")
        shape = Shape(self.shape)
        object.__setattr__(self, "_shape", shape)
        object.__setattr__(self, "shape_spec", shape.to_json())
        object.__setattr__(self, "max_shape", shape.max_shape)
        object.__setattr__(self, "dynamic", shape.dynamic)
        object.__setattr__(self, "dtype", normalize_dtype(self.dtype))

    @property
    def rank(self) -> int:
        return self._shape.rank

    @property
    def numel(self) -> int:
        return self._shape.numel

    @property
    def constraints(self) -> list[dict[str, Any]]:
        return self._shape.constraints


class Parameter:
    def __init__(
        self,
        shape: Sequence[int | Dim | Mapping[str, Any]] | Shape | np.ndarray | Any,
        dtype: str = "float32",
        name: Optional[str] = None,
        value: Any = None,
    ):
        if value is None and _looks_like_value(shape):
            array = np.asarray(shape)
            dtype = str(array.dtype)
            self._shape = Shape(array.shape)
            self.shape_spec = self._shape.to_json()
            self.shape = self._shape.max_shape
            self.dtype = normalize_dtype(dtype)
            self._value = _normalize_constant_value(array, self.dtype, self.shape)
        else:
            self._shape = Shape(shape)
            self.shape_spec = self._shape.to_json()
            self.shape = self._shape.max_shape
            self.dtype = normalize_dtype(dtype)
            self._value = None if value is None else _normalize_constant_value(value, self.dtype, self.shape)
        self.name = name

    @property
    def value(self) -> np.ndarray | None:
        return self._value

    def bind(self, value: Any) -> "Parameter":
        return Parameter(self.shape_spec, dtype=self.dtype, name=self.name, value=value)

    @property
    def rank(self) -> int:
        return self._shape.rank

    @property
    def dynamic(self) -> bool:
        return self._shape.dynamic

    @property
    def numel(self) -> int:
        return self._shape.numel


class Tensor:
    def __init__(
        self,
        name: str,
        shape: Sequence[int],
        dtype: str,
        builder: "GraphBuilder",
        kind: str = "intermediate",
        shape_spec: Sequence[int | Mapping[str, Any]] | None = None,
    ):
        self.name = name
        self._shape = Shape(shape_spec or shape)
        self._shape.validate_max_shape(shape)
        self.shape = self._shape.max_shape
        self.shape_spec = self._shape.to_json()
        self.dtype = normalize_dtype(dtype)
        self.builder = builder
        self.kind = kind
        self.output_name: Optional[str] = None

    @property
    def rank(self) -> int:
        return self._shape.rank

    @property
    def dynamic(self) -> bool:
        return self._shape.dynamic

    @property
    def numel(self) -> int:
        return self._shape.numel

    def __add__(self, other: Any) -> "Tensor":
        from dinoml import ops

        return ops.add(self, other)

    def __radd__(self, other: Any) -> "Tensor":
        from dinoml import ops

        return ops.add(other, self)

    def __sub__(self, other: Any) -> "Tensor":
        from dinoml import ops

        return ops.sub(self, other)

    def __rsub__(self, other: Any) -> "Tensor":
        from dinoml import ops

        return ops.sub(other, self)

    def __mul__(self, other: Any) -> "Tensor":
        from dinoml import ops

        return ops.mul(self, other)

    def __rmul__(self, other: Any) -> "Tensor":
        from dinoml import ops

        return ops.mul(other, self)

    def __truediv__(self, other: Any) -> "Tensor":
        from dinoml import ops

        return ops.div(self, other)

    def __rtruediv__(self, other: Any) -> "Tensor":
        from dinoml import ops

        return ops.div(other, self)


class Module:
    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter) and value.name is None:
            value.name = name
        super().__setattr__(name, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__}.forward is not implemented")


class GraphBuilder:
    def __init__(self, name: str):
        self.name = name
        self.nodes: List[Dict[str, Any]] = []
        self.tensors: Dict[str, Dict[str, Any]] = {}
        self.inputs: List[Dict[str, Any]] = []
        self.states: List[Dict[str, Any]] = []
        self.constants: List[Dict[str, Any]] = []
        self.constant_values: Dict[str, np.ndarray] = {}
        self.views: List[Dict[str, Any]] = []
        self.output_shape_report_tensors: set[str] = set()
        self._constant_tensors: Dict[Parameter, Tensor] = {}
        self._next_tensor_id = 0
        self._next_node_id = 0

    def __enter__(self) -> "GraphBuilder":
        global _CURRENT_BUILDER
        if _CURRENT_BUILDER is not None:
            raise RuntimeError("Nested DinoML tracing is not supported")
        _CURRENT_BUILDER = self
        return self

    def __exit__(self, *_exc: Any) -> None:
        global _CURRENT_BUILDER
        _CURRENT_BUILDER = None

    @staticmethod
    def current() -> "GraphBuilder":
        if _CURRENT_BUILDER is None:
            raise RuntimeError("DinoML ops can only be used while tracing")
        return _CURRENT_BUILDER

    def input(self, name: str, spec: TensorSpec) -> Tensor:
        if name in self.tensors:
            raise ValueError(f"Duplicate tensor name: {name}")
        tensor = Tensor(name=name, shape=spec.max_shape, dtype=spec.dtype, builder=self, kind="input", shape_spec=spec.shape_spec)
        info = _tensor_info(tensor)
        self.inputs.append(_io_info(name, tensor))
        self.tensors[name] = info
        return tensor

    def state(self, name: str, spec: TensorSpec) -> Tensor:
        if name in self.tensors:
            raise ValueError(f"Duplicate tensor name: {name}")
        tensor = Tensor(name=name, shape=spec.max_shape, dtype=spec.dtype, builder=self, kind="state", shape_spec=spec.shape_spec)
        info = _tensor_info(tensor)
        self.states.append(_state_info(name, tensor))
        self.tensors[name] = info
        return tensor

    def constant(self, parameter: Parameter) -> Tensor:
        if parameter in self._constant_tensors:
            return self._constant_tensors[parameter]
        base_name = parameter.name or f"constant_{len(self.constants)}"
        name = self._unique_name(base_name)
        tensor = Tensor(
            name=name,
            shape=parameter.shape,
            dtype=parameter.dtype,
            builder=self,
            kind="constant",
            shape_spec=parameter.shape_spec,
        )
        nbytes = int(shape_numel(tensor.shape) * dtype_nbytes(tensor.dtype))
        self.constants.append(
            {
                "name": name,
                "tensor": name,
                "shape": tensor.shape,
                "shape_spec": tensor.shape_spec,
                "layout": dense_layout(tensor.shape),
                "dtype": tensor.dtype,
                "offset": None,
                "nbytes": nbytes,
            }
        )
        self.tensors[name] = _tensor_info(tensor)
        if parameter.value is not None:
            self.constant_values[name] = _normalize_constant_value(parameter.value, tensor.dtype, tensor.shape)
        self._constant_tensors[parameter] = tensor
        return tensor

    def emit(
        self,
        op: str,
        inputs: Sequence[Tensor],
        shape: Sequence[int],
        dtype: str,
        attrs: Optional[Dict[str, Any]] = None,
        shape_spec: Sequence[int | Mapping[str, Any]] | None = None,
    ) -> Tensor:
        output_name = self._new_tensor_name()
        tensor = Tensor(output_name, shape=shape, dtype=dtype, builder=self, shape_spec=shape_spec)
        node_id = f"n{self._next_node_id}"
        self._next_node_id += 1
        self.nodes.append(
            {
                "id": node_id,
                "op": op,
                "inputs": [tensor.name for tensor in inputs],
                "outputs": [output_name],
                "attrs": attrs or {},
            }
        )
        self.tensors[output_name] = _tensor_info(tensor)
        return tensor

    def emit_multi(
        self,
        op: str,
        inputs: Sequence[Tensor],
        outputs: Sequence[tuple[Sequence[int], str, Sequence[int | Mapping[str, Any]] | None]],
        attrs: Optional[Dict[str, Any]] = None,
    ) -> tuple[Tensor, ...]:
        output_tensors = []
        output_names = []
        for shape, dtype, shape_spec in outputs:
            output_name = self._new_tensor_name()
            tensor = Tensor(output_name, shape=shape, dtype=dtype, builder=self, shape_spec=shape_spec)
            output_tensors.append(tensor)
            output_names.append(output_name)
            self.tensors[output_name] = _tensor_info(tensor)
        node_id = f"n{self._next_node_id}"
        self._next_node_id += 1
        self.nodes.append(
            {
                "id": node_id,
                "op": op,
                "inputs": [tensor.name for tensor in inputs],
                "outputs": output_names,
                "attrs": attrs or {},
            }
        )
        return tuple(output_tensors)

    def emit_view(
        self,
        transform: str,
        source: Tensor,
        shape: Sequence[int],
        shape_spec: Sequence[int | Mapping[str, Any]],
    ) -> Tensor:
        output_name = self._new_tensor_name()
        tensor = Tensor(output_name, shape=shape, dtype=source.dtype, builder=self, shape_spec=shape_spec)
        self.tensors[output_name] = _tensor_info(tensor)
        self.views.append(
            {
                "tensor": output_name,
                "source": source.name,
                "kind": "shape_view",
                "transform": transform,
                "offset_elements": 0,
                "shape": list(tensor.shape),
                "shape_spec": list(tensor.shape_spec),
            }
        )
        return tensor

    def report_output_shape(self, tensor: Tensor) -> None:
        if tensor.builder is not self:
            raise ValueError("Cannot register an output shape report for a tensor from another DinoML trace")
        self.output_shape_report_tensors.add(str(tensor.name))

    def to_ir(self, outputs: Sequence[Tensor]) -> Dict[str, Any]:
        output_infos = []
        aliased_output_tensors: set[str] = set()
        input_tensors = {item["tensor"] for item in self.inputs}
        state_tensors = {item["tensor"] for item in self.states}
        constant_tensors = {item["tensor"] for item in self.constants}
        view_tensors = {view["tensor"] for view in self.views}
        for idx, tensor in enumerate(outputs):
            output_name = tensor.output_name or f"output_{idx}"
            output_tensor = tensor
            if tensor.name in input_tensors or tensor.name in state_tensors or tensor.name in constant_tensors or tensor.name in aliased_output_tensors:
                if tensor.name in view_tensors:
                    raise NotImplementedError("Duplicate public outputs of view aliases are not supported yet")
                output_tensor = self._add_public_output_alias(tensor)
            aliased_output_tensors.add(output_tensor.name)
            output_infos.append(
                {
                    "name": output_name,
                    "tensor": output_tensor.name,
                    "shape": output_tensor.shape,
                    "shape_spec": output_tensor.shape_spec,
                    "layout": dense_layout(output_tensor.shape),
                    "dtype": output_tensor.dtype,
                }
            )
            self.tensors[output_tensor.name]["kind"] = "output"

        metadata = _shape_metadata([*self.inputs, *self.states, *self.constants, *output_infos, *self.tensors.values()])
        if self.views:
            metadata["views"] = {"version": VIEW_METADATA_VERSION, "views": self.views}
        if self.output_shape_report_tensors:
            reports = []
            for output in output_infos:
                if str(output["tensor"]) not in self.output_shape_report_tensors:
                    continue
                reports.append({"output": str(output["name"]), "kind": "shape_buffer"})
            if reports:
                metadata["output_shape_reports"] = {
                    "version": OUTPUT_SHAPE_REPORT_METADATA_VERSION,
                    "reports": reports,
                }

        return {
            "schema_version": IR_SCHEMA_VERSION,
            "name": self.name,
            "inputs": self.inputs,
            "states": self.states,
            "constants": self.constants,
            "outputs": output_infos,
            "nodes": self.nodes,
            "tensors": list(self.tensors.values()),
            "metadata": metadata,
        }

    def _add_public_output_alias(self, source: Tensor) -> Tensor:
        output_name = self._new_tensor_name()
        tensor = Tensor(output_name, shape=source.shape, dtype=source.dtype, builder=self, shape_spec=source.shape_spec)
        self.tensors[output_name] = _tensor_info(tensor)
        self.views.append(
            {
                "tensor": output_name,
                "source": source.name,
                "kind": "shape_view",
                "transform": "identity",
                "offset_elements": 0,
                "shape": list(tensor.shape),
                "shape_spec": list(tensor.shape_spec),
            }
        )
        return tensor

    def _new_tensor_name(self) -> str:
        while True:
            name = f"t{self._next_tensor_id}"
            self._next_tensor_id += 1
            if name not in self.tensors:
                return name

    def _unique_name(self, base_name: str) -> str:
        candidate = base_name
        idx = 1
        while candidate in self.tensors:
            candidate = f"{base_name}_{idx}"
            idx += 1
        return candidate


def _tensor_info(tensor: Tensor) -> Dict[str, Any]:
    info = {
        "name": tensor.name,
        "shape": list(tensor.shape),
        "shape_spec": list(tensor.shape_spec),
        "layout": dense_layout(tensor.shape),
        "dtype": tensor.dtype,
        "kind": tensor.kind,
        "nbytes": int(shape_numel(tensor.shape) * dtype_nbytes(tensor.dtype)),
    }
    return info


def _io_info(name: str, tensor: Tensor) -> Dict[str, Any]:
    return {
        "name": name,
        "tensor": tensor.name,
        "shape": tensor.shape,
        "shape_spec": tensor.shape_spec,
        "layout": dense_layout(tensor.shape),
        "dtype": tensor.dtype,
    }


def _state_info(name: str, tensor: Tensor) -> Dict[str, Any]:
    return {
        "name": name,
        "tensor": tensor.name,
        "shape": tensor.shape,
        "shape_spec": tensor.shape_spec,
        "layout": dense_layout(tensor.shape),
        "dtype": tensor.dtype,
        "nbytes": int(shape_numel(tensor.shape) * dtype_nbytes(tensor.dtype)),
    }


def state(name: str, spec: TensorSpec) -> Tensor:
    return GraphBuilder.current().state(name, spec)


def as_tensor(value: Any, dtype_hint: str | None = None) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, Parameter):
        return GraphBuilder.current().constant(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        dtype = normalize_dtype(dtype_hint or "float32")
        return GraphBuilder.current().constant(Parameter([], dtype=dtype, value=value))
    raise TypeError(f"Expected DinoML Tensor or Parameter, got {type(value).__name__}")


def trace(
    model: Module,
    inputs: Dict[str, TensorSpec],
    name: Optional[str] = None,
    constants: Mapping[str, Any] | None = None,
) -> ModelSpec:
    graph_name = name or type(model).__name__
    with GraphBuilder(graph_name) as builder:
        input_tensors = {input_name: builder.input(input_name, spec) for input_name, spec in inputs.items()}
        outputs = model(**input_tensors)
        output_tensors = list(_flatten_outputs(outputs))
        ir = builder.to_ir(output_tensors)
    spec = ModelSpec(name=graph_name, ir=ir, constants=builder.constant_values)
    if constants:
        spec = spec.bind_constants(constants)
    return spec


def _flatten_outputs(outputs: Any) -> Iterable[Tensor]:
    if isinstance(outputs, Tensor):
        yield outputs
    elif isinstance(outputs, (list, tuple)):
        for output in outputs:
            yield from _flatten_outputs(output)
    elif isinstance(outputs, dict):
        for name, output in outputs.items():
            for tensor in _flatten_outputs(output):
                tensor.output_name = str(name)
                yield tensor
    else:
        raise TypeError(f"Unsupported model output type: {type(outputs).__name__}")


def _looks_like_value(value: Any) -> bool:
    if isinstance(value, Shape):
        return False
    if isinstance(value, np.ndarray):
        return True
    if isinstance(value, (list, tuple)):
        return not all(
            isinstance(dim, (int, Dim)) or (isinstance(dim, Mapping) and dim.get("kind") in {"dim", "int_expr"})
            for dim in value
        )
    return not isinstance(value, Sequence)


def _normalize_constant_value(value: Any, dtype: str, shape: Sequence[int]) -> np.ndarray:
    array = array_to_storage(value, dtype)
    expected_shape = tuple(int(dim) for dim in shape)
    if array.shape != expected_shape:
        raise ValueError(f"Constant value has shape {array.shape}, expected {expected_shape}")
    return array


def _shape_metadata(items: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    constraints = []
    seen = set()
    for item in items:
        shape_spec = item.get("shape_spec")
        if shape_spec is None:
            continue
        for constraint in shape_constraints(shape_spec):
            key = (item.get("tensor"), constraint["axis"], constraint["name"])
            if key in seen:
                continue
            seen.add(key)
            enriched = dict(constraint)
            enriched["tensor"] = item.get("tensor")
            constraints.append(enriched)
    return {
        "dynamic_shapes": bool(constraints),
        "shape_constraints": constraints,
        "allocation_semantics": "max_shape_static_buffers",
    }
