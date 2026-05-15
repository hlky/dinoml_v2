from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from dinoml.shapes import symbolic_int_interval


def shape_buffer_context(tensor: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ident": c_ident(str(tensor["name"])),
        "rank": len(tensor["shape"]),
        "shape_literal": shape_literal(tensor["shape"]),
    }


def dynamic_dim_sources(
    *,
    input_map: Mapping[str, int],
    output_map: Mapping[str, int],
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    sources: dict[str, str] = {}
    for tensor_name, idx in input_map.items():
        for axis, dim in enumerate(tensor_map[tensor_name].get("shape_spec", tensor_map[tensor_name]["shape"])):
            _record_direct_dim_source(sources, dim, f"inputs[{idx}].shape[{axis}]")
    for tensor_name, idx in output_map.items():
        for axis, dim in enumerate(tensor_map[tensor_name].get("shape_spec", tensor_map[tensor_name]["shape"])):
            _record_direct_dim_source(sources, dim, f"outputs[{idx}].shape[{axis}]")
    return sources


def shape_dim_expr(tensor: Mapping[str, Any], axis: int, dynamic_dims: Mapping[str, str]) -> str:
    dim = tensor.get("shape_spec", tensor["shape"])[axis]
    return shape_spec_dim_expr(dim, dynamic_dims)


def shape_spec_dim_expr(dim: Any, dynamic_dims: Mapping[str, str]) -> str:
    if isinstance(dim, int):
        return str(int(dim))
    kind = dim.get("kind")
    if kind == "dim":
        return dynamic_dims.get(str(dim["name"]), str(int(dim["max"])))
    if kind != "int_expr":
        raise ValueError(f"Unsupported shape dimension kind: {kind!r}")
    op = str(dim["op"])
    lhs = shape_spec_dim_expr(dim["lhs"], dynamic_dims)
    rhs = shape_spec_dim_expr(dim["rhs"], dynamic_dims)
    if op == "add":
        return f"({lhs} + {rhs})"
    if op == "sub":
        return f"({lhs} - {rhs})"
    if op == "mul":
        return f"({lhs} * {rhs})"
    if op == "div":
        return f"dinoml::module::floor_div({lhs}, {rhs})"
    raise ValueError(f"Unsupported shape expression op: {op!r}")


def validate_symbolic_int_sources(
    *,
    items: Iterable[Mapping[str, Any]],
    dynamic_dims: Mapping[str, str],
    context: str,
) -> None:
    for item in items:
        item_name = str(item.get("name", item.get("tensor", "<unknown>")))
        for axis, dim in enumerate(item.get("shape_spec", item["shape"])):
            if not isinstance(dim, Mapping) or dim.get("kind") != "int_expr":
                continue
            missing = sorted(
                {str(leaf["name"]) for leaf in named_dim_leaves(dim) if str(leaf["name"]) not in dynamic_dims}
            )
            if missing:
                raise NotImplementedError(
                    "Symbolic integer shape expressions require direct runtime sources for all named dimensions "
                    f"({context} entry {item_name!r}, axis {axis}, missing {missing})."
                )


def shape_dim_range(dim: Any) -> dict[str, int]:
    if isinstance(dim, int):
        return {"min": int(dim), "max": int(dim), "divisible_by": 1}
    if dim.get("kind") == "int_expr":
        min_dim, max_dim = symbolic_int_interval(dim)
        return {"min": int(min_dim), "max": int(max_dim), "divisible_by": 1}
    return {
        "min": int(dim["min"]),
        "max": int(dim["max"]),
        "divisible_by": int(dim.get("divisible_by", 1)),
    }


def named_dim_leaves(dim: Any) -> list[dict[str, Any]]:
    if isinstance(dim, int):
        return []
    kind = dim.get("kind")
    if kind == "dim":
        return [dict(dim)]
    if kind == "int_expr":
        return [*named_dim_leaves(dim["lhs"]), *named_dim_leaves(dim["rhs"])]
    raise ValueError(f"Unsupported shape dimension kind: {kind!r}")


def expression_axis_checks(
    *,
    items: Iterable[Mapping[str, Any]],
    array_name: str,
    dynamic_dims: Mapping[str, str],
) -> list[str]:
    checks: list[str] = []
    for tensor_idx, item in enumerate(items):
        for axis, dim in enumerate(item.get("shape_spec", item["shape"])):
            if not isinstance(dim, Mapping) or dim.get("kind") != "int_expr":
                continue
            actual_expr = f"{array_name}[{tensor_idx}].shape[{axis}]"
            expected_expr = shape_spec_dim_expr(dim, dynamic_dims)
            checks.append(
                f'  if ({actual_expr} != {expected_expr}) return dinoml::module::fail("Shape expression mismatch for {item["name"]} axis {axis}");'
            )
    return checks


def constant_expression_axis_checks(
    *,
    constants: Iterable[Mapping[str, Any]],
    dynamic_dims: Mapping[str, str],
    c_ident_fn: Any,
) -> list[str]:
    checks: list[str] = []
    for item in constants:
        ident = c_ident_fn(str(item["tensor"]))
        for axis, dim in enumerate(item.get("shape_spec", item["shape"])):
            if not isinstance(dim, Mapping) or dim.get("kind") != "int_expr":
                continue
            actual_expr = f"session->module->const_shape_{ident}[{axis}]"
            expected_expr = shape_spec_dim_expr(dim, dynamic_dims)
            checks.append(
                f'  if ({actual_expr} != {expected_expr}) return dinoml::module::fail("Shape expression mismatch for {item["name"]} axis {axis}");'
            )
    return checks


def numel_expr(ident: str, rank: int) -> str:
    if rank == 0:
        return "1"
    return " * ".join(f"shape_{ident}_{axis}" for axis in range(rank))


def shape_vars_literal(ident: str, rank: int) -> str:
    return ", ".join(f"shape_{ident}_{axis}" for axis in range(rank))


def shape_literal(shape: Iterable[int]) -> str:
    return ", ".join(str(int(dim)) for dim in shape)


def c_ident(name: str) -> str:
    ident = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not ident or ident[0].isdigit():
        ident = f"_{ident}"
    ident = re.sub(r"_(\d+)$", r"__\1", ident)
    return ident


def _record_direct_dim_source(sources: dict[str, str], dim: Any, expr: str) -> None:
    if isinstance(dim, int):
        return
    if dim.get("kind") == "dim":
        sources.setdefault(str(dim["name"]), expr)
