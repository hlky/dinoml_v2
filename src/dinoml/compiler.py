from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from dinoml.backends.registry import get_backend_spec
from dinoml.backends.target import Target
from dinoml.ir import (
    ARTIFACT_SCHEMA_VERSION,
    RUNTIME_ABI_VERSION,
    ModelSpec,
    array_to_storage,
    graph_hash,
    read_json,
    write_json,
)
from dinoml.kernels.manifest import apply_execution_plan, build_kernel_manifest
from dinoml.kernels.codegen import create_codegen_plan
from dinoml.kernels.profiling import profile_artifact
from dinoml.ops.definitions import get_op_def
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
    execution_plan: str | Path | Mapping[str, Any] | None = None,
    profile: bool = False,
    profile_iterations: int = 20,
    profile_input_shapes: Mapping[str, Any] | None = None,
    profile_refresh: bool = False,
) -> Artifact:
    if profile:
        if execution_plan is not None:
            raise ValueError("compile(profile=True) cannot also consume an explicit execution_plan")
        if target.name != "cuda":
            raise ValueError("compile(profile=True) currently supports CUDA targets only")
        return _compile_with_profile(
            spec,
            target,
            output,
            clean=clean,
            pass_manager=pass_manager,
            iterations=profile_iterations,
            input_shapes=profile_input_shapes,
            refresh=profile_refresh,
        )
    return _compile_once(
        spec,
        target,
        output,
        clean=clean,
        pass_manager=pass_manager,
        execution_plan=execution_plan,
    )


def _compile_with_profile(
    spec: ModelSpec,
    target: Target,
    output: str | Path,
    *,
    clean: bool,
    pass_manager: Optional[PassManager],
    iterations: int,
    input_shapes: Mapping[str, Any] | None,
    refresh: bool,
) -> Artifact:
    initial_artifact = _compile_once(
        spec,
        target,
        output,
        clean=clean,
        pass_manager=pass_manager,
        execution_plan=None,
    )
    profile_report = profile_artifact(
        initial_artifact.path,
        input_shapes=input_shapes,
        iterations=iterations,
        refresh=refresh,
    )
    execution_plan_summary = profile_report.get("execution_plan", {})
    if not isinstance(execution_plan_summary, Mapping) or not execution_plan_summary.get("path"):
        raise ValueError("Profiler did not produce an execution plan")
    if int(execution_plan_summary.get("selection_count", 0) or 0) == 0:
        write_json(initial_artifact.path / "debug" / "bootstrap_profile_report.json", dict(profile_report))
        return initial_artifact
    execution_plan_payload = read_json(Path(str(execution_plan_summary["path"])))
    final_artifact = _compile_once(
        spec,
        target,
        output,
        clean=True,
        pass_manager=pass_manager,
        execution_plan=execution_plan_payload,
    )
    write_json(final_artifact.path / "debug" / "bootstrap_profile_report.json", dict(profile_report))
    return final_artifact


def _compile_once(
    spec: ModelSpec,
    target: Target,
    output: str | Path,
    *,
    clean: bool = True,
    pass_manager: Optional[PassManager] = None,
    execution_plan: str | Path | Mapping[str, Any] | None = None,
) -> Artifact:
    backend = get_backend_spec(target.name)
    execution_plan_payload = _load_execution_plan(execution_plan)
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
    if execution_plan_payload is not None:
        compile_config["execution_plan"] = _execution_plan_compile_config(execution_plan_payload)

    write_json(artifact_dir / "graph.dinoir.json", lowered_ir)
    write_json(artifact_dir / "metadata.json", _runtime_metadata(lowered_ir))
    kernel_manifest = build_kernel_manifest(lowered_ir, target.to_json())
    if execution_plan_payload is not None:
        _validate_execution_plan_overlay(execution_plan_payload, target.to_json(), kernel_manifest)
        kernel_manifest = apply_execution_plan(kernel_manifest, execution_plan_payload, strict=True)
        write_json(debug_dir / "execution_plan.json", dict(execution_plan_payload))
    write_json(artifact_dir / "compile_config.json", compile_config)
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
    if _requires_kernel_library(kernel_manifest, "cutlass_gemm"):
        files["cutlass_gemm_library"] = "lib/libdinoml_cutlass_gemm.so"

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


def _load_execution_plan(execution_plan: str | Path | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if execution_plan is None:
        return None
    if isinstance(execution_plan, Mapping):
        return dict(execution_plan)
    return read_json(Path(execution_plan))


def _validate_execution_plan_overlay(
    execution_plan: Mapping[str, Any],
    target: Mapping[str, Any],
    kernel_manifest: Mapping[str, Any],
) -> None:
    kind = execution_plan.get("kind")
    if kind is not None and kind != "dinoml.execution_plan":
        raise ValueError(f"Unsupported execution plan kind: {kind!r}")
    schema_version = execution_plan.get("schema_version")
    if schema_version is not None and int(schema_version) != 1:
        raise ValueError(f"Unsupported execution plan schema version: {schema_version!r}")
    plan_target = execution_plan.get("target")
    if isinstance(plan_target, Mapping) and dict(plan_target) and dict(plan_target) != dict(target):
        raise ValueError(
            f"Execution plan target {dict(plan_target)!r} does not match compile target {dict(target)!r}"
        )
    expected_manifest_key = execution_plan.get("kernel_manifest_cache_key")
    if expected_manifest_key and expected_manifest_key != kernel_manifest["cache_key"]:
        raise ValueError(
            "Execution plan was generated for a different kernel manifest "
            f"({expected_manifest_key} != {kernel_manifest['cache_key']})"
        )
    if not execution_plan.get("static_selections") and not execution_plan.get("selections"):
        raise ValueError("Execution plan does not contain any candidate selections to apply")


def _execution_plan_compile_config(execution_plan: Mapping[str, Any]) -> dict[str, Any]:
    summary = dict(execution_plan.get("summary", {})) if isinstance(execution_plan.get("summary"), Mapping) else {}
    return {
        "schema_version": execution_plan.get("schema_version"),
        "execution_plan_key": execution_plan.get("execution_plan_key"),
        "kernel_manifest_cache_key": execution_plan.get("kernel_manifest_cache_key"),
        "selection_policy": execution_plan.get("selection_policy"),
        "static_selection_policy": execution_plan.get("static_selection_policy"),
        "summary": summary,
    }


def _runtime_metadata(ir: Dict) -> Dict:
    return {
        "runtime_abi_version": RUNTIME_ABI_VERSION,
        "name": ir["name"],
        "inputs": ir["inputs"],
        "outputs": ir["outputs"],
        "constants": ir["constants"],
        "memory_plan": ir.get("metadata", {}).get("memory_plan", {}),
    }


def _requires_kernel_library(kernel_manifest: Dict, library: str) -> bool:
    return any(item.get("kernel_library") == library for item in kernel_manifest.get("required_kernels", []))


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
    tensor_map = {tensor["name"]: tensor for tensor in ir["tensors"]}
    for node in ir["nodes"]:
        op_def = get_op_def(str(node["op"]))
        if target.name not in op_def.backend_kernels:
            raise NotImplementedError(f"{target.name} backend does not support op {op_def.name}")
        node_tensor_names = [*node.get("inputs", []), *node.get("outputs", [])]
        node_dtypes = sorted({tensor_map[name]["dtype"] for name in node_tensor_names if name in tensor_map})
        unsupported_node_dtypes = [dtype for dtype in node_dtypes if dtype not in op_def.allowed_dtypes]
        if unsupported_node_dtypes:
            raise NotImplementedError(
                f"Op {op_def.name} supports dtypes {list(op_def.allowed_dtypes)}; "
                f"unsupported compiled dtypes: {unsupported_node_dtypes}"
            )
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
