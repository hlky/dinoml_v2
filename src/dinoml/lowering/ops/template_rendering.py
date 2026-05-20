from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.target_specs import LoweringTargetSpec, lowering_target_spec


def render_op_template(name: str, context: Mapping[str, Any]) -> str:
    return _template_env().get_template(name).render(**context)


def supported_target_spec(target: str, op_name: str) -> LoweringTargetSpec:
    try:
        spec = lowering_target_spec(target)
    except ValueError as exc:
        raise ValueError(f"Unsupported {op_name} target: {target}") from exc
    if not spec.generated_module_admitted:
        raise ValueError(f"Unsupported {op_name} target: {target}")
    return spec


@lru_cache(maxsize=1)
def _template_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
