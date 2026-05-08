from __future__ import annotations

import re
from typing import Any, Iterable, Mapping


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
            if isinstance(dim, int):
                continue
            sources.setdefault(str(dim["name"]), f"inputs[{idx}].shape[{axis}]")
    for tensor_name, idx in output_map.items():
        for axis, dim in enumerate(tensor_map[tensor_name].get("shape_spec", tensor_map[tensor_name]["shape"])):
            if isinstance(dim, int):
                continue
            sources.setdefault(str(dim["name"]), f"outputs[{idx}].shape[{axis}]")
    return sources


def shape_dim_expr(tensor: Mapping[str, Any], axis: int, dynamic_dims: Mapping[str, str]) -> str:
    dim = tensor.get("shape_spec", tensor["shape"])[axis]
    if isinstance(dim, int):
        return str(int(dim))
    return dynamic_dims.get(str(dim["name"]), str(int(dim["max"])))


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
    return ident
