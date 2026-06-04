from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from dinoml.ir import VIEW_METADATA_VERSION
from dinoml.passes.utils import tensor_map
from dinoml.passes.validation import validate_view_metadata

TEMPORARY_ARENA_ALIGNMENT = 64


@dataclass(frozen=True)
class TemporaryLifetime:
    tensor: str
    nbytes: int
    aligned_nbytes: int
    first_use_node: int
    last_use_node: int


@dataclass(frozen=True)
class PlannedTemporary(TemporaryLifetime):
    offset: int


def plan_temporary_memory(ir: Mapping[str, Any]) -> dict[str, Any]:
    output_tensors = {str(output["tensor"]) for output in ir.get("outputs", [])}
    input_tensors = {str(item["tensor"]) for item in ir.get("inputs", [])}
    state_tensors = {str(item["tensor"]) for item in ir.get("states", [])}
    constant_tensors = {str(item["tensor"]) for item in ir.get("constants", [])}
    tensors = tensor_map(ir)
    views = validate_view_metadata(ir.get("metadata", {}).get("views"), tensors)
    view_tensors = {str(view["tensor"]) for view in views}
    view_sources = {str(view["tensor"]): str(view["source"]) for view in views}

    temporary_names: list[str] = []
    temporary_sizes: dict[str, int] = {}
    for tensor in ir.get("tensors", []):
        name = str(tensor["name"])
        if (
            name in output_tensors
            or name in input_tensors
            or name in state_tensors
            or name in constant_tensors
            or name in view_tensors
        ):
            continue
        temporary_names.append(name)
        temporary_sizes[name] = int(tensor.get("nbytes", 0) or 0)

    lifetimes = _temporary_lifetimes(
        temporary_names=temporary_names,
        temporary_sizes=temporary_sizes,
        nodes=ir.get("nodes", []),
        output_tensors=output_tensors,
        view_sources=view_sources,
    )
    planned = _allocate_temporary_offsets(lifetimes)

    return {
        "allocation": "lifetime_planned_temporaries",
        "planner": "greedy_by_size",
        "alignment": TEMPORARY_ARENA_ALIGNMENT,
        "temporaries": [
            {
                "tensor": item.tensor,
                "nbytes": item.nbytes,
                "aligned_nbytes": item.aligned_nbytes,
                "offset": item.offset,
                "first_use_node": item.first_use_node,
                "last_use_node": item.last_use_node,
            }
            for item in planned
        ],
        "views": {"version": VIEW_METADATA_VERSION, "views": views},
        "arena_nbytes": _arena_nbytes(planned),
        "workspace_nbytes": _arena_nbytes(planned),
        "total_temporary_nbytes": sum(item.nbytes for item in planned),
    }


def _temporary_lifetimes(
    *,
    temporary_names: Sequence[str],
    temporary_sizes: Mapping[str, int],
    nodes: Sequence[Mapping[str, Any]],
    output_tensors: set[str],
    view_sources: Mapping[str, str],
) -> list[TemporaryLifetime]:
    num_nodes = len(nodes)
    first_default = 0 if num_nodes == 0 else num_nodes
    lifetimes = {
        name: {
            "tensor": name,
            "nbytes": int(temporary_sizes[name]),
            "aligned_nbytes": _align_nbytes(int(temporary_sizes[name])),
            "first_use_node": first_default,
            "last_use_node": -1,
        }
        for name in temporary_names
    }

    def touch(name: Any, node_index: int) -> None:
        tensor_name = str(name)
        owner_name = str(view_sources.get(tensor_name, tensor_name))
        item = lifetimes.get(owner_name)
        if item is None:
            return
        item["first_use_node"] = min(int(item["first_use_node"]), node_index)
        item["last_use_node"] = max(int(item["last_use_node"]), node_index)

    for node_index, node in enumerate(nodes):
        for tensor_name in node.get("inputs", []):
            touch(tensor_name, node_index)
        for tensor_name in node.get("outputs", []):
            touch(tensor_name, node_index)

    output_lifetime_node = 0 if num_nodes == 0 else num_nodes - 1
    for output_name in output_tensors:
        source_name = view_sources.get(output_name)
        if source_name is None:
            continue
        item = lifetimes.get(str(source_name))
        if item is None:
            continue
        item["last_use_node"] = max(int(item["last_use_node"]), output_lifetime_node)

    result: list[TemporaryLifetime] = []
    for name in temporary_names:
        item = lifetimes[name]
        if int(item["last_use_node"]) < 0:
            item["first_use_node"] = 0
            item["last_use_node"] = 0
        elif int(item["first_use_node"]) > int(item["last_use_node"]):
            item["first_use_node"] = int(item["last_use_node"])
        result.append(
            TemporaryLifetime(
                tensor=str(item["tensor"]),
                nbytes=int(item["nbytes"]),
                aligned_nbytes=int(item["aligned_nbytes"]),
                first_use_node=int(item["first_use_node"]),
                last_use_node=int(item["last_use_node"]),
            )
        )
    return result


def _allocate_temporary_offsets(lifetimes: Sequence[TemporaryLifetime]) -> list[PlannedTemporary]:
    ordered = sorted(
        lifetimes,
        key=lambda item: (-item.aligned_nbytes, item.first_use_node, item.tensor),
    )
    assigned: list[PlannedTemporary] = []
    planned_by_name: dict[str, PlannedTemporary] = {}
    for item in ordered:
        previous_end = 0
        best_offset: int | None = None
        smallest_gap: int | None = None
        for allocated in assigned:
            if _intervals_overlap(item, allocated):
                gap = allocated.offset - previous_end
                if gap >= item.aligned_nbytes and (smallest_gap is None or gap < smallest_gap):
                    best_offset = previous_end
                    smallest_gap = gap
                previous_end = max(previous_end, allocated.offset + allocated.aligned_nbytes)
        if best_offset is None:
            best_offset = previous_end
        planned = PlannedTemporary(
            tensor=item.tensor,
            nbytes=item.nbytes,
            aligned_nbytes=item.aligned_nbytes,
            first_use_node=item.first_use_node,
            last_use_node=item.last_use_node,
            offset=best_offset,
        )
        planned_by_name[planned.tensor] = planned

        insert_at = len(assigned)
        for index, allocated in enumerate(assigned):
            if allocated.offset > planned.offset:
                insert_at = index
                break
        assigned.insert(insert_at, planned)
    return [planned_by_name[item.tensor] for item in lifetimes]


def _intervals_overlap(left: TemporaryLifetime, right: TemporaryLifetime) -> bool:
    return max(left.first_use_node, right.first_use_node) <= min(left.last_use_node, right.last_use_node)


def _align_nbytes(nbytes: int) -> int:
    if nbytes <= 0:
        return 0
    return ((nbytes + TEMPORARY_ARENA_ALIGNMENT - 1) // TEMPORARY_ARENA_ALIGNMENT) * TEMPORARY_ARENA_ALIGNMENT


def _arena_nbytes(planned: Sequence[PlannedTemporary]) -> int:
    return max((item.offset + item.aligned_nbytes for item in planned), default=0)
