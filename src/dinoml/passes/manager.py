from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from dinoml.fusions.elementwise import elementwise_fusion
from dinoml.ir import graph_hash, write_json
from dinoml.passes.core import backend_lower, canonicalize, constant_bind, dead_code_eliminate, memory_plan, shape_type_infer
from dinoml.passes.validation import validate_ir


@dataclass(frozen=True)
class PassReport:
    name: str
    before_hash: str
    after_hash: str

    @property
    def changed(self) -> bool:
        return self.before_hash != self.after_hash


PassFn = Callable[[Dict[str, Any]], Dict[str, Any]]


class PassManager:
    DEFAULT_PIPELINE: Sequence[str] = (
        "canonicalize",
        "shape_type_infer",
        "constant_bind",
        "dead_code_eliminate",
        "elementwise_fusion",
        "memory_plan",
        "backend_lower",
    )

    def __init__(self, pipeline: Optional[Sequence[str]] = None):
        self.pipeline = tuple(pipeline or self.DEFAULT_PIPELINE)
        self._registry: Dict[str, PassFn] = {
            "canonicalize": canonicalize,
            "shape_type_infer": shape_type_infer,
            "constant_bind": constant_bind,
            "dead_code_eliminate": dead_code_eliminate,
            "elementwise_fusion": elementwise_fusion,
            "memory_plan": memory_plan,
            "backend_lower": backend_lower,
        }

    def run(self, ir: Mapping[str, Any], dump_dir: Optional[Path] = None) -> tuple[Dict[str, Any], List[PassReport]]:
        current = copy.deepcopy(dict(ir))
        reports: List[PassReport] = []
        if dump_dir is not None:
            dump_dir.mkdir(parents=True, exist_ok=True)
            write_json(dump_dir / "00_initial.json", current)
        validate_ir(current)
        for idx, pass_name in enumerate(self.pipeline, start=1):
            if pass_name not in self._registry:
                raise ValueError(f"Unknown pass: {pass_name}")
            before = graph_hash(current)
            current = self._registry[pass_name](copy.deepcopy(current))
            validate_ir(current)
            after = graph_hash(current)
            reports.append(PassReport(pass_name, before, after))
            if dump_dir is not None:
                write_json(dump_dir / f"{idx:02d}_{pass_name}.json", current)
        return current, reports
