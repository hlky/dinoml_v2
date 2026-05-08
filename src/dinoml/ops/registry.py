from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence


ShapeFn = Callable[[Sequence[Sequence[int]]], list[int]]


@dataclass(frozen=True)
class KernelVariant:
    symbol: str
    profiler_symbol: str | None = None
    candidates: tuple[Mapping[str, Any], ...] = ()
    candidate_set: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class KernelBinding:
    symbol: str
    library: str
    profiler_symbol: str | None = None
    source_template: str | None = None
    candidates: tuple[Mapping[str, Any], ...] = ()
    candidate_set: Mapping[str, Any] | None = None
    dtype_variants: Mapping[str, KernelVariant] = field(default_factory=dict)

    def resolve(self, dtype: str | None = None) -> KernelBinding:
        if not self.dtype_variants or dtype is None:
            return self
        try:
            variant = self.dtype_variants[dtype]
        except KeyError as exc:
            supported = ", ".join(sorted(self.dtype_variants))
            raise ValueError(f"Kernel binding does not support dtype {dtype!r}; supported dtypes: {supported}") from exc
        return KernelBinding(
            symbol=variant.symbol,
            library=self.library,
            profiler_symbol=variant.profiler_symbol,
            source_template=self.source_template,
            candidates=variant.candidates,
            candidate_set=variant.candidate_set,
        )


@dataclass(frozen=True)
class AttrDef:
    name: str
    type_name: str
    default: Any = None
    required: bool = False


@dataclass(frozen=True)
class OpSchema:
    inputs: tuple[str, ...] = ()
    attrs: tuple[AttrDef, ...] = ()


@dataclass(frozen=True)
class FrontendBinding:
    name: str
    default_attrs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OpDef:
    name: str
    schema: OpSchema
    infer_shape: ShapeFn
    allowed_dtypes: tuple[str, ...] = ("float32",)
    backend_kernels: Mapping[str, KernelBinding] = field(default_factory=dict)
    frontend: FrontendBinding | None = None
    profiler: bool = False
    variadic_inputs: bool = False
    description: str = ""

    @property
    def input_count(self) -> int:
        return len(self.schema.inputs)

    def accepts_input_count(self, input_count: int) -> bool:
        if self.variadic_inputs:
            return input_count >= self.input_count
        return input_count == self.input_count


class OpRegistry:
    def __init__(self) -> None:
        self.definitions: Dict[str, OpDef] = {}
        self.frontends: Dict[str, str] = {}

    def register(self, op_def: OpDef) -> OpDef:
        if op_def.name in self.definitions:
            raise ValueError(f"Duplicate op registration: {op_def.name}")
        if op_def.frontend is not None:
            if op_def.frontend.name in self.frontends:
                existing = self.frontends[op_def.frontend.name]
                raise ValueError(f"Duplicate frontend op registration: {op_def.frontend.name} ({existing}, {op_def.name})")
            self.frontends[op_def.frontend.name] = op_def.name
        self.definitions[op_def.name] = op_def
        return op_def

    def get(self, op_name: str) -> OpDef:
        try:
            return self.definitions[op_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported op: {op_name}") from exc

    def get_frontend(self, frontend_name: str) -> OpDef:
        try:
            op_name = self.frontends[frontend_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported frontend op: {frontend_name}") from exc
        return self.get(op_name)

    def frontend_names(self) -> list[str]:
        return sorted(self.frontends)

    def op_defs(self) -> Iterable[OpDef]:
        return self.definitions.values()
