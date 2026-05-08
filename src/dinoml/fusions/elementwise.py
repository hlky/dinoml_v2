from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Set

from dinoml.ops.elementwise import FUSABLE_ELEMENTWISE_OPS
from dinoml.passes.utils import consumer_map, tensor_map


def elementwise_fusion(ir: Dict[str, Any]) -> Dict[str, Any]:
    tensors = tensor_map(ir)
    producer = {output: node for node in ir["nodes"] for output in node["outputs"]}
    consumers = consumer_map(ir["nodes"])
    node_order = {node["id"]: idx for idx, node in enumerate(ir["nodes"])}
    fusable = {node["id"]: node for node in ir["nodes"] if node["op"] in FUSABLE_ELEMENTWISE_OPS}
    visited: Set[str] = set()
    replacements: Dict[str, Dict[str, Any]] = {}
    nodes_to_remove: Set[str] = set()
    fusion_groups = []

    for node in ir["nodes"]:
        if node["id"] not in fusable or node["id"] in visited:
            continue

        component_ids = _collect_fusable_component(node, fusable, producer, consumers, tensors)
        visited.update(component_ids)
        component = sorted((fusable[node_id] for node_id in component_ids), key=lambda item: node_order[item["id"]])
        external_outputs = _fused_external_outputs(component, component_ids, consumers, tensors)
        if not external_outputs:
            continue
        output_shapes = {tuple(tensors[name]["shape"]) for name in external_outputs}
        if len(output_shapes) != 1:
            continue
        external_inputs = _fused_external_inputs(component, component_ids, producer)
        if not external_inputs:
            continue

        fused_id = "_".join(node["id"] for node in component) + "_fused"
        anchor_id = component[-1]["id"]
        replacements[anchor_id] = {
            "id": fused_id,
            "op": "fused_elementwise",
            "inputs": external_inputs,
            "outputs": external_outputs,
            "attrs": {
                "sub_ops": [
                    {
                        "id": item["id"],
                        "op": item["op"],
                        "inputs": list(item["inputs"]),
                        "outputs": list(item["outputs"]),
                        "attrs": dict(item.get("attrs", {})),
                    }
                    for item in component
                ],
            },
        }
        nodes_to_remove.update(component_ids)
        fusion_groups.append([item["id"] for item in component])

    if not replacements:
        ir.setdefault("metadata", {})["fusion_groups"] = []
        return ir

    new_nodes = []
    for node in ir["nodes"]:
        if node["id"] in replacements:
            new_nodes.append(replacements[node["id"]])
        elif node["id"] not in nodes_to_remove:
            new_nodes.append(node)

    fused_outputs = {output for node in replacements.values() for output in node["outputs"]}
    fused_inputs = {input_name for node in replacements.values() for input_name in node["inputs"]}
    fused_intermediate_tensors = set()
    for node in replacements.values():
        for sub_op in node["attrs"]["sub_ops"]:
            for output in sub_op["outputs"]:
                if output not in fused_outputs and output not in fused_inputs:
                    fused_intermediate_tensors.add(output)
    ir["nodes"] = new_nodes
    ir["tensors"] = [tensor for tensor in ir["tensors"] if tensor["name"] not in fused_intermediate_tensors]
    ir.setdefault("metadata", {})["fusion_groups"] = fusion_groups
    return ir


def _collect_fusable_component(
    root: Mapping[str, Any],
    fusable: Mapping[str, Mapping[str, Any]],
    producer: Mapping[str, Mapping[str, Any]],
    consumers: Mapping[str, list[Mapping[str, Any]]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> Set[str]:
    component: Set[str] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node["id"] in component:
            continue
        component.add(node["id"])
        for input_name in node["inputs"]:
            parent = producer.get(input_name)
            if parent is not None and _can_cross_fused_edge(input_name, parent, fusable, consumers, tensors):
                stack.append(parent)
        for output_name in node["outputs"]:
            if tensors[output_name]["kind"] == "output":
                continue
            output_consumers = consumers.get(output_name, [])
            if output_consumers and all(consumer["id"] in fusable for consumer in output_consumers):
                stack.extend(output_consumers)
    return component


def _can_cross_fused_edge(
    tensor_name: str,
    parent: Mapping[str, Any],
    fusable: Mapping[str, Mapping[str, Any]],
    consumers: Mapping[str, list[Mapping[str, Any]]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> bool:
    if parent["id"] not in fusable:
        return False
    if tensors[tensor_name]["kind"] == "output":
        return False
    return all(consumer["id"] in fusable for consumer in consumers.get(tensor_name, []))


def _fused_external_outputs(
    component: Sequence[Mapping[str, Any]],
    component_ids: Set[str],
    consumers: Mapping[str, list[Mapping[str, Any]]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    outputs = []
    for node in component:
        for output in node["outputs"]:
            output_consumers = consumers.get(output, [])
            if tensors[output]["kind"] == "output" or any(consumer["id"] not in component_ids for consumer in output_consumers):
                outputs.append(output)
    return list(dict.fromkeys(outputs))


def _fused_external_inputs(
    component: Sequence[Mapping[str, Any]],
    component_ids: Set[str],
    producer: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    inputs = []
    for node in component:
        for input_name in node["inputs"]:
            parent = producer.get(input_name)
            if parent is None or parent["id"] not in component_ids:
                inputs.append(input_name)
    return list(dict.fromkeys(inputs))
