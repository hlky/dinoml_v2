from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping


def tensor_map(ir: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {tensor["name"]: dict(tensor) for tensor in ir["tensors"]}


def consumer_map(nodes: Iterable[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    consumers: Dict[str, List[Mapping[str, Any]]] = {}
    for node in nodes:
        for input_name in node["inputs"]:
            consumers.setdefault(input_name, []).append(node)
    return consumers
