from __future__ import annotations

from math import prod
from typing import Any, Dict, Mapping, Sequence, Set

from dinoml.ir import VIEW_METADATA_VERSION, dtype_nbytes
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


def add_layer_norm_fusion(ir: Dict[str, Any]) -> Dict[str, Any]:
    tensors = tensor_map(ir)
    nodes = ir["nodes"]
    fused_nodes = []
    fusion_groups = []
    idx = 0
    while idx < len(nodes):
        node = nodes[idx]
        if idx + 1 < len(nodes) and _is_single_add_fused_elementwise(node):
            next_node = nodes[idx + 1]
            add_sub_op = node["attrs"]["sub_ops"][0]
            add_inputs = list(add_sub_op["inputs"])
            if (
                next_node["op"] == "layer_norm"
                and next_node["inputs"][0] == node["outputs"][0]
                and len(add_inputs) == 2
                and list(tensors[add_inputs[0]]["shape"]) == list(tensors[add_inputs[1]]["shape"])
            ):
                fused_nodes.append(
                    {
                        "id": f"{node['id']}_{next_node['id']}_add_layer_norm_fused",
                        "op": "add_layer_norm",
                        "inputs": [
                            add_inputs[0],
                            add_inputs[1],
                            next_node["inputs"][1],
                            next_node["inputs"][2],
                        ],
                        "outputs": [
                            node["outputs"][0],
                            next_node["outputs"][0],
                        ],
                        "attrs": {"eps": next_node.get("attrs", {}).get("eps", 1e-5)},
                    }
                )
                fusion_groups.append([node["id"], next_node["id"]])
                idx += 2
                continue
        fused_nodes.append(node)
        idx += 1
    if fusion_groups:
        ir["nodes"] = fused_nodes
    ir.setdefault("metadata", {})["add_layer_norm_fusion_groups"] = fusion_groups
    return ir


def sliced_add_layer_norm_fusion(ir: Dict[str, Any]) -> Dict[str, Any]:
    tensors = tensor_map(ir)
    producer = {output: node for node in ir["nodes"] for output in node["outputs"]}
    consumers = consumer_map(ir["nodes"])
    views = [dict(view) for view in ir.get("metadata", {}).get("views", {}).get("views", [])]
    view_by_tensor = {str(view["tensor"]): view for view in views}
    views_by_source: dict[str, list[dict[str, Any]]] = {}
    for view in views:
        views_by_source.setdefault(str(view["source"]), []).append(view)
    output_tensors = {str(output["tensor"]) for output in ir.get("outputs", [])}
    existing_names = {str(tensor["name"]) for tensor in ir["tensors"]}

    remove_node_ids: set[str] = set()
    replacements: dict[str, dict[str, Any]] = {}
    remove_tensors: set[str] = set()
    added_tensors: list[dict[str, Any]] = []
    added_views: list[dict[str, Any]] = []
    removed_view_tensors: set[str] = set()
    fusion_groups: list[list[str]] = []

    for node in ir["nodes"]:
        if node.get("op") != "layer_norm" or len(node.get("inputs", [])) != 3:
            continue
        layer_norm_input = str(node["inputs"][0])
        view = view_by_tensor.get(layer_norm_input)
        if view is None:
            continue
        add_output = str(view["source"])
        add_node = producer.get(add_output)
        if add_node is None or not _is_single_add_fused_elementwise(add_node):
            continue
        if consumers.get(add_output) or add_output in output_tensors:
            continue
        if any(str(source_view["tensor"]) != layer_norm_input for source_view in views_by_source.get(add_output, [])):
            continue
        add_sub_op = add_node["attrs"]["sub_ops"][0]
        add_inputs = [str(input_name) for input_name in add_sub_op.get("inputs", [])]
        if len(add_inputs) != 2:
            continue
        if list(tensors[add_inputs[0]]["shape"]) != list(tensors[add_inputs[1]]["shape"]):
            continue
        if list(tensors[add_output]["shape"]) != list(tensors[add_inputs[0]]["shape"]):
            continue

        view_shape = [int(dim) for dim in view["shape"]]
        view_shape_spec = list(view.get("shape_spec", view_shape))
        offset = int(view.get("offset_elements", 0))
        input_views = []
        for input_name in add_inputs:
            view_name = _fresh_name(f"{layer_norm_input}_{input_name}_slice", existing_names)
            input_views.append(view_name)
            source_tensor = tensors[input_name]
            added_tensors.append(
                {
                    "name": view_name,
                    "kind": "temporary",
                    "shape": view_shape,
                    "shape_spec": view_shape_spec,
                    "dtype": source_tensor["dtype"],
                    "nbytes": int(prod(view_shape)) * dtype_nbytes(str(source_tensor["dtype"])),
                }
            )
            added_views.append(
                {
                    "tensor": view_name,
                    "source": input_name,
                    "kind": "shape_view",
                    "transform": str(view.get("transform", "dynamic_slice")),
                    "offset_elements": offset,
                    "shape": view_shape,
                    "shape_spec": view_shape_spec,
                }
            )

        summed_name = _fresh_name(f"{layer_norm_input}_summed", existing_names)
        output_tensor = tensors[str(node["outputs"][0])]
        added_tensors.append(
            {
                "name": summed_name,
                "kind": "temporary",
                "shape": list(output_tensor["shape"]),
                "shape_spec": list(output_tensor.get("shape_spec", output_tensor["shape"])),
                "dtype": output_tensor["dtype"],
                "nbytes": int(output_tensor["nbytes"]),
            }
        )
        replacements[str(node["id"])] = {
            "id": f"{add_node['id']}_{node['id']}_sliced_add_layer_norm_fused",
            "op": "add_layer_norm",
            "inputs": [input_views[0], input_views[1], node["inputs"][1], node["inputs"][2]],
            "outputs": [summed_name, node["outputs"][0]],
            "attrs": {"eps": node.get("attrs", {}).get("eps", 1e-5)},
        }
        remove_node_ids.add(str(add_node["id"]))
        remove_tensors.update({add_output, layer_norm_input})
        removed_view_tensors.add(layer_norm_input)
        fusion_groups.append([str(add_node["id"]), str(node["id"])])

    if not replacements:
        ir.setdefault("metadata", {})["sliced_add_layer_norm_fusion_groups"] = []
        return ir

    ir["nodes"] = [
        replacements[str(node["id"])]
        if str(node["id"]) in replacements
        else node
        for node in ir["nodes"]
        if str(node["id"]) not in remove_node_ids
    ]
    ir["tensors"] = [
        tensor
        for tensor in ir["tensors"]
        if str(tensor["name"]) not in remove_tensors
    ]
    ir["tensors"].extend(added_tensors)
    kept_views = [
        view
        for view in views
        if str(view["tensor"]) not in removed_view_tensors and str(view["source"]) not in remove_tensors
    ]
    kept_views.extend(added_views)
    metadata = ir.setdefault("metadata", {})
    metadata["views"] = {"version": VIEW_METADATA_VERSION, "views": kept_views}
    metadata.pop("memory_plan", None)
    metadata["sliced_add_layer_norm_fusion_groups"] = fusion_groups
    return ir


def _fresh_name(prefix: str, existing_names: set[str]) -> str:
    candidate = prefix
    idx = 0
    while candidate in existing_names:
        idx += 1
        candidate = f"{prefix}_{idx}"
    existing_names.add(candidate)
    return candidate


def _is_single_add_fused_elementwise(node: Mapping[str, Any]) -> bool:
    if node.get("op") != "fused_elementwise":
        return False
    sub_ops = node.get("attrs", {}).get("sub_ops", [])
    return (
        len(sub_ops) == 1
        and sub_ops[0].get("op") == "add"
        and list(sub_ops[0].get("outputs", [])) == list(node.get("outputs", []))
    )


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
