from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.cpp_types import cpu_storage_type, cuda_storage_type
from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.gather import _index_storage_type
from dinoml.ops.collections import GATHER_INDEX_DTYPES
from dinoml.ops.embedding import EMBEDDING_DTYPES, resolve_embedding_shape


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    context = _context(target, node, tensor_map)
    if target == "cpu":
        return _render_template("embedding_cpu.cpp.j2", context)
    if target == "cuda":
        return _render_template("embedding_cuda.cu.j2", context)
    raise ValueError(f"Unsupported embedding target: {target}")


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    func = _function_name(node, tensor_map)
    table = _c_ident(node["inputs"][0])
    indices = _c_ident(node["inputs"][1])
    out = _c_ident(node["outputs"][0])
    args = f"ptr_{table}, ptr_{indices}, ptr_{out}, runtime_numel_{indices}, runtime_numel_{out}"
    if target == "cpu":
        return f"if (int err = {func}({args})) return err;"
    if target == "cuda":
        return f"if (int err = {func}({args}, session->stream)) return err;"
    raise ValueError(f"Unsupported embedding target: {target}")


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    if target not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported embedding target: {target}")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    table_tensor = tensor_map[node["inputs"][0]]
    index_tensor = tensor_map[node["inputs"][1]]
    output_tensor = tensor_map[node["outputs"][0]]
    _validate_node_contract(node, table_tensor, index_tensor, output_tensor)
    dtype = str(output_tensor["dtype"])
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": cpu_storage_type(dtype) if target == "cpu" else cuda_storage_type(dtype),
        "index_storage_type": _index_storage_type(str(index_tensor["dtype"])),
        "vocab_size": int(table_tensor["shape"][0]),
        "hidden_size": int(table_tensor["shape"][1]),
        "copy_body": _copy_body(table_tensor, target),
        "block_size": 256,
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    table_tensor: Mapping[str, Any],
    index_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> None:
    if node["op"] != "embedding":
        raise ValueError(f"Unsupported embedding op: {node['op']}")
    if len(node.get("inputs", [])) != 2:
        raise ValueError("embedding expects two tensor inputs")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("embedding expects exactly one output")
    dtype = str(output_tensor["dtype"])
    if dtype not in EMBEDDING_DTYPES:
        raise NotImplementedError(f"embedding lowering does not support dtype {output_tensor['dtype']}")
    if str(table_tensor["dtype"]) != dtype:
        raise ValueError("embedding table and output dtype must match")
    if str(index_tensor["dtype"]) not in GATHER_INDEX_DTYPES:
        raise ValueError(f"embedding indices must have dtype int64 or int32, got {index_tensor['dtype']}")
    if len(table_tensor["shape"]) != 2:
        raise ValueError("embedding lowering requires rank-2 table")
    if len(index_tensor["shape"]) < 1:
        raise ValueError("embedding lowering requires rank >= 1 indices")
    table_shape_spec = table_tensor.get("shape_spec", table_tensor["shape"])
    if any(not isinstance(dim, int) for dim in table_shape_spec):
        raise ValueError("embedding lowering requires a static table shape [vocab, hidden]")
    expected_shape = resolve_embedding_shape(table_tensor["shape"], index_tensor["shape"])
    if list(expected_shape) != list(output_tensor["shape"]):
        raise ValueError("embedding output shape does not match indices shape plus hidden size")


def _copy_body(table_tensor: Mapping[str, Any], target: str) -> str:
    vocab_size = int(table_tensor["shape"][0])
    hidden_size = int(table_tensor["shape"][1])
    lines = [
        f"  const int64_t row = idx / {hidden_size};",
        f"  const int64_t hidden_offset = idx - row * {hidden_size};",
        "  const int64_t selected_index = static_cast<int64_t>(indices[row]);",
        f"  if (selected_index < 0 || selected_index >= {vocab_size}) {{",
    ]
    if target == "cpu":
        lines.append('    return dino_runtime_fail("embedding index out of bounds");')
    else:
        lines.append(f"    assert(selected_index >= 0 && selected_index < {vocab_size});")
        lines.append("    return;")
    lines.extend(
        [
            "  }",
            f"  const int64_t table_idx = selected_index * {hidden_size} + hidden_offset;",
            "  y[idx] = table[table_idx];",
        ]
    )
    return "\n".join(lines)


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    table_tensor = tensor_map[node["inputs"][0]]
    index_tensor = tensor_map[node["inputs"][1]]
    output_tensor = tensor_map[node["outputs"][0]]
    signature = {
        "op": "embedding",
        "table_shape": list(table_tensor["shape"]),
        "index_shape": list(index_tensor["shape"]),
        "output_shape": list(output_tensor["shape"]),
        "dtype": str(output_tensor["dtype"]),
        "index_dtype": str(index_tensor["dtype"]),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"embedding_{digest}"


def _render_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


def _c_ident(name: str) -> str:
    ident = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not ident or ident[0].isdigit():
        ident = f"_{ident}"
    ident = re.sub(r"_(\d+)$", r"__\1", ident)
    return ident


EMBEDDING_LOWERING = OpLowering(
    op_name="embedding",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
