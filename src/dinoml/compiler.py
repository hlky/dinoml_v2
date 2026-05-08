from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from dinoml.backends.registry import get_backend_spec
from dinoml.backends.target import Target
from dinoml.ir import (
    ARTIFACT_SCHEMA_VERSION,
    RUNTIME_ABI_VERSION,
    ModelSpec,
    array_to_storage,
    canonical_json,
    graph_hash,
    write_json,
)
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.kernels.codegen import create_codegen_plan
from dinoml.passes import PassManager


@dataclass(frozen=True)
class Artifact:
    path: Path


def compile(
    spec: ModelSpec,
    target: Target,
    output: str | Path,
    *,
    clean: bool = True,
    pass_manager: Optional[PassManager] = None,
) -> Artifact:
    backend = get_backend_spec(target.name)
    artifact_dir = Path(output).resolve()
    if clean and artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = artifact_dir / "debug"
    pass_dump_dir = debug_dir / "pass_dumps"
    generated_src_dir = debug_dir / "generated_src"
    generated_src_dir.mkdir(parents=True, exist_ok=True)

    manager = pass_manager or PassManager()
    lowered_ir, reports = manager.run(spec.ir, dump_dir=pass_dump_dir)
    _validate_mvp_runtime_contract(lowered_ir, target)
    lowered_ir = _write_constants(artifact_dir, lowered_ir, spec.constants)

    compile_config = {
        "target": target.to_json(),
        "passes": [
            {
                "name": report.name,
                "before_hash": report.before_hash,
                "after_hash": report.after_hash,
                "changed": report.changed,
            }
            for report in reports
        ],
    }

    write_json(artifact_dir / "graph.dinoir.json", lowered_ir)
    write_json(artifact_dir / "metadata.json", _runtime_metadata(lowered_ir))
    write_json(artifact_dir / "compile_config.json", compile_config)
    kernel_manifest = build_kernel_manifest(lowered_ir, target.to_json())
    write_json(artifact_dir / "kernel_manifest.json", kernel_manifest)
    codegen_plan = create_codegen_plan(
        kernel_manifest,
        Path(os.environ.get("DINOML_CACHE_DIR", Path.home() / ".cache" / "dinoml_v2")),
    )
    write_json(artifact_dir / "kernel_codegen_plan.json", codegen_plan.to_json())

    files = {
        "graph": "graph.dinoir.json",
        "module": "module.so",
        "metadata": "metadata.json",
        "constants": "constants.bin",
        "compile_config": "compile_config.json",
        "kernel_manifest": "kernel_manifest.json",
        "kernel_codegen_plan": "kernel_codegen_plan.json",
    }
    files.update(backend.support_libraries)

    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "runtime_abi_version": RUNTIME_ABI_VERSION,
        "name": spec.name,
        "target": target.to_json(),
        "files": files,
        "graph_hash": graph_hash(lowered_ir),
    }
    write_json(artifact_dir / "manifest.json", manifest)

    backend.resolve_build_function()(
        lowered_ir,
        target=target,
        artifact_dir=artifact_dir,
        generated_src_dir=generated_src_dir,
        kernel_manifest=kernel_manifest,
    )

    return Artifact(artifact_dir)


def _runtime_metadata(ir: Dict) -> Dict:
    return {
        "runtime_abi_version": RUNTIME_ABI_VERSION,
        "name": ir["name"],
        "inputs": ir["inputs"],
        "outputs": ir["outputs"],
        "constants": ir["constants"],
        "memory_plan": ir.get("metadata", {}).get("memory_plan", {}),
    }


def _write_constants(artifact_dir: Path, ir: Dict, constants: Dict[str, np.ndarray]) -> Dict:
    offset = 0
    constant_infos = []
    with (artifact_dir / "constants.bin").open("wb") as handle:
        for constant in ir["constants"]:
            name = constant["name"]
            if name not in constants:
                raise ValueError(f"Missing constant value: {name}")
            array = array_to_storage(constants[name], constant["dtype"])
            expected_shape = tuple(int(dim) for dim in constant["shape"])
            if array.shape != expected_shape:
                raise ValueError(f"Constant {name} has shape {array.shape}, expected {expected_shape}")
            data = array.tobytes(order="C")
            constant = dict(constant)
            constant["offset"] = offset
            constant["nbytes"] = len(data)
            constant_infos.append(constant)
            handle.write(data)
            offset += len(data)
    ir = dict(ir)
    ir["constants"] = constant_infos
    ir.setdefault("metadata", {})["constants_nbytes"] = offset
    return ir


def _validate_mvp_runtime_contract(ir: Dict, target: Target) -> None:
    dtypes = {tensor["dtype"] for tensor in ir["tensors"]}
    supported = get_backend_spec(target.name).supported_dtypes
    unsupported = sorted(dtype for dtype in dtypes if dtype not in supported)
    if unsupported:
        raise NotImplementedError(
            f"The current {target.name} runtime supports dtypes {sorted(supported)}; "
            f"unsupported compiled dtypes: {unsupported}"
        )
    views = ir.get("metadata", {}).get("memory_plan", {}).get("views", {}).get("views", [])
    view_tensors = {view["tensor"] for view in views}
    view_sources = {view["source"] for view in views}
    view_of_view = sorted(view_tensors & view_sources)
    if view_of_view:
        raise NotImplementedError(
            "View-of-view aliases are not supported by the current runtime lowering; "
            f"view tensors used as view sources: {view_of_view}"
        )
    node_view_outputs = sorted(
        output_name
        for node in ir["nodes"]
        for output_name in node["outputs"]
        if output_name in view_tensors
    )
    if node_view_outputs:
        raise NotImplementedError(
            "View alias tensors cannot be written by kernels; metadata.views must describe "
            f"shape-only aliases of an owning tensor. Kernel outputs using view storage: {node_view_outputs}"
        )
