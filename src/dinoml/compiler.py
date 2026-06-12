from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from dinoml.backends.registry import get_backend_spec
from dinoml.backends.target import Target
from dinoml.ir import (
    ARTIFACT_SCHEMA_VERSION,
    RUNTIME_ABI_VERSION,
    ModelSpec,
    canonical_json,
    dtype_nbytes,
    dtype_numpy,
    graph_hash,
    read_json,
    write_json,
)
from dinoml.constant_sources import (
    GGUFConstant,
    GGUF_MATERIALIZATION_DEQUANTIZE_FULL_BEFORE_LAUNCH,
    GGUF_MATERIALIZATION_DEQUANTIZE_ON_GPU_BEFORE_LAUNCH,
    GGUF_RESIDENCY_EAGER_DENSE_DEVICE,
    GGUF_RESIDENCY_MANUAL_RUNTIME_LOAD,
    gguf_constant_policy_status,
    materialize_gguf_encoded_constant,
    materialize_constant_value,
)
from dinoml.kernels.manifest import apply_execution_plan, build_kernel_manifest
from dinoml.kernels.codegen import create_codegen_plan
from dinoml.kernels.profiling import profile_artifact
from dinoml.lowering.ops.conv import render_conv_wrapper_source
from dinoml.lowering.shape_buffers import validate_symbolic_int_sources
from dinoml.ops.collections import INDEX_ADD_DTYPES
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
    profile_repeats: int = 3,
    profile_input_shapes: Mapping[str, Any] | None = None,
    profile_refresh: bool = False,
    cutlass_conv_validation_mode: str = "fast",
    constant_load_policy: str = "eager",
) -> Artifact:
    constant_load_policy = _validate_constant_load_policy(constant_load_policy)
    if profile:
        if execution_plan is not None:
            raise ValueError("compile(profile=True) cannot also consume an explicit execution_plan")
        if target.name not in {"cuda", "rocm"}:
            raise ValueError("compile(profile=True) currently supports CUDA and ROCm targets only")
        return _compile_with_profile(
            spec,
            target,
            output,
            clean=clean,
            pass_manager=pass_manager,
            iterations=profile_iterations,
            repeats=profile_repeats,
            input_shapes=profile_input_shapes,
            refresh=profile_refresh,
            cutlass_conv_validation_mode=cutlass_conv_validation_mode,
            constant_load_policy=constant_load_policy,
        )
    return _compile_once(
        spec,
        target,
        output,
        clean=clean,
        pass_manager=pass_manager,
        execution_plan=execution_plan,
        constant_load_policy=constant_load_policy,
    )


def _compile_with_profile(
    spec: ModelSpec,
    target: Target,
    output: str | Path,
    *,
    clean: bool,
    pass_manager: Optional[PassManager],
    iterations: int,
    repeats: int,
    input_shapes: Mapping[str, Any] | None,
    refresh: bool,
    cutlass_conv_validation_mode: str = "fast",
    constant_load_policy: str = "eager",
) -> Artifact:
    backend = get_backend_spec(target.name)
    artifact_dir = _prepare_artifact_dir(output, clean=clean)
    debug_dir = artifact_dir / "debug"
    generated_src_dir = debug_dir / "generated_src"
    generated_src_dir.mkdir(parents=True, exist_ok=True)
    profile_artifact_dir = artifact_dir
    lowered_ir, reports = _lower_for_compile(
        spec,
        target,
        artifact_dir=artifact_dir,
        pass_manager=pass_manager,
    )
    _validate_profile_shape_expressions(lowered_ir, target)
    kernel_manifest = build_kernel_manifest(lowered_ir, target.to_json())
    previous_cutlass_conv_profiler_env = os.environ.get("DINOML_BUILD_CUTLASS_CONV_PROFILERS")
    if target.name == "cuda":
        os.environ["DINOML_BUILD_CUTLASS_CONV_PROFILERS"] = "1"
    try:
        _materialize_profile_bootstrap_artifact(
            spec,
            target,
            artifact_dir=artifact_dir,
            lowered_ir=lowered_ir,
            reports=reports,
            backend=backend,
            constant_load_policy=constant_load_policy,
            kernel_manifest=kernel_manifest,
        )
        profile_report = profile_artifact(
            profile_artifact_dir,
            input_shapes=input_shapes,
            iterations=iterations,
            repeats=repeats,
            refresh=refresh,
            cutlass_conv_validation_mode=cutlass_conv_validation_mode,
        )
    finally:
        if target.name == "cuda":
            if previous_cutlass_conv_profiler_env is None:
                os.environ.pop("DINOML_BUILD_CUTLASS_CONV_PROFILERS", None)
            else:
                os.environ["DINOML_BUILD_CUTLASS_CONV_PROFILERS"] = previous_cutlass_conv_profiler_env
    execution_plan_summary = profile_report.get("execution_plan", {})
    if not isinstance(execution_plan_summary, Mapping) or not execution_plan_summary.get("path"):
        raise ValueError("Profiler did not produce an execution plan")
    execution_plan_payload = (
        read_json(Path(str(execution_plan_summary["path"])))
        if int(execution_plan_summary.get("selection_count", 0) or 0) != 0
        else None
    )
    _reset_generated_sources(generated_src_dir)
    final_artifact = _build_artifact_from_lowered_ir(
        spec,
        target,
        artifact_dir=artifact_dir,
        generated_src_dir=generated_src_dir,
        lowered_ir=lowered_ir,
        reports=reports,
        backend=backend,
        execution_plan_payload=execution_plan_payload,
        constant_load_policy=constant_load_policy,
    )
    _materialize_bootstrap_profile_report(final_artifact.path)
    return final_artifact


def _materialize_profile_bootstrap_artifact(
    spec: ModelSpec,
    target: Target,
    *,
    artifact_dir: Path,
    lowered_ir: Mapping[str, Any],
    reports: Sequence[Any],
    backend: Any,
    constant_load_policy: str,
    kernel_manifest: Mapping[str, Any] | None = None,
) -> None:
    debug_dir = artifact_dir / "debug"
    lowered_ir = dict(lowered_ir)
    write_json(artifact_dir / "graph.dinoir.json", lowered_ir)
    write_json(artifact_dir / "metadata.json", _runtime_metadata(lowered_ir))
    encoded_constants_manifest = _encoded_constants_manifest(lowered_ir)
    if encoded_constants_manifest is not None:
        write_json(artifact_dir / "encoded_constants.json", encoded_constants_manifest)
    else:
        (artifact_dir / "encoded_constants.json").unlink(missing_ok=True)
    kernel_manifest = build_kernel_manifest(lowered_ir, target.to_json()) if kernel_manifest is None else dict(kernel_manifest)
    _validate_gguf_runtime_dequant_admission(lowered_ir, target, kernel_manifest)
    write_json(artifact_dir / "kernel_manifest.json", kernel_manifest)
    codegen_plan = create_codegen_plan(
        kernel_manifest,
        Path(os.environ.get("DINOML_CACHE_DIR", Path.home() / ".cache" / "dinoml_v2")),
    )
    write_json(artifact_dir / "kernel_codegen_plan.json", codegen_plan.to_json())
    if target.name == "rocm":
        from dinoml.backends.rocm import ensure_rocm_support_libs

        ensure_rocm_support_libs(target.arch, kernel_manifest=kernel_manifest)
    files = {
        "graph": "graph.dinoir.json",
        "metadata": "metadata.json",
        "constants": "constants.bin",
        "compile_config": "compile_config.json",
        "kernel_manifest": "kernel_manifest.json",
        "kernel_codegen_plan": "kernel_codegen_plan.json",
    }
    files.update(backend.support_libraries)
    if encoded_constants_manifest is not None:
        files["encoded_constants"] = "encoded_constants.json"
    _copy_profile_support_libraries(target, artifact_dir, kernel_manifest, files)
    write_json(
        artifact_dir / "compile_config.json",
        {
            "target": target.to_json(),
            "constant_load_policy": constant_load_policy,
            "passes": [
                {
                    "name": report.name,
                    "before_hash": report.before_hash,
                    "after_hash": report.after_hash,
                    "changed": report.changed,
                }
                for report in reports
            ],
        },
    )
    write_json(
        artifact_dir / "manifest.json",
        {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "runtime_abi_version": RUNTIME_ABI_VERSION,
            "name": spec.name,
            "target": target.to_json(),
            "constant_load_policy": constant_load_policy,
            "files": files,
            "graph_hash": graph_hash(lowered_ir),
        },
    )


def _copy_profile_support_libraries(
    target: Target,
    artifact_dir: Path,
    kernel_manifest: Mapping[str, Any],
    files: dict[str, str],
) -> None:
    if target.name != "cuda":
        return
    from dinoml.backends.cuda import ensure_cuda_support_libs

    support_libs = ensure_cuda_support_libs(target.arch, kernel_manifest=kernel_manifest)
    artifact_lib_dir = artifact_dir / "lib"
    artifact_lib_dir.mkdir(parents=True, exist_ok=True)
    for source in (support_libs.runtime_lib, support_libs.cuda_runtime_lib, support_libs.kernels_lib):
        shutil.copy2(source, artifact_lib_dir / source.name)


def _compile_once(
    spec: ModelSpec,
    target: Target,
    output: str | Path,
    *,
    clean: bool = True,
    pass_manager: Optional[PassManager] = None,
    execution_plan: str | Path | Mapping[str, Any] | None = None,
    constant_load_policy: str = "eager",
) -> Artifact:
    backend = get_backend_spec(target.name)
    execution_plan_payload = _load_execution_plan(execution_plan)
    artifact_dir = _prepare_artifact_dir(output, clean=clean)
    debug_dir = artifact_dir / "debug"
    generated_src_dir = debug_dir / "generated_src"
    generated_src_dir.mkdir(parents=True, exist_ok=True)
    lowered_ir, reports = _lower_for_compile(
        spec,
        target,
        artifact_dir=artifact_dir,
        pass_manager=pass_manager,
    )
    return _build_artifact_from_lowered_ir(
        spec,
        target,
        artifact_dir=artifact_dir,
        generated_src_dir=generated_src_dir,
        lowered_ir=lowered_ir,
        reports=reports,
        backend=backend,
        execution_plan_payload=execution_plan_payload,
        constant_load_policy=constant_load_policy,
    )


def _prepare_artifact_dir(output: str | Path, *, clean: bool) -> Path:
    artifact_dir = Path(output).resolve()
    if clean and artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _reset_generated_sources(generated_src_dir: Path) -> None:
    if generated_src_dir.exists():
        shutil.rmtree(generated_src_dir)
    generated_src_dir.mkdir(parents=True, exist_ok=True)


def _materialize_bootstrap_profile_report(artifact_dir: Path) -> None:
    debug_dir = artifact_dir / "debug"
    source = debug_dir / "profile_report.json"
    target = debug_dir / "bootstrap_profile_report.json"
    if not source.exists():
        raise FileNotFoundError(f"Expected profiler report at {source}")
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _lower_for_compile(
    spec: ModelSpec,
    target: Target,
    *,
    artifact_dir: Path,
    pass_manager: Optional[PassManager],
) -> tuple[dict[str, Any], Sequence[Any]]:
    manager = pass_manager or PassManager()
    lowered_ir, reports = manager.run(spec.ir, dump_dir=artifact_dir / "debug" / "pass_dumps")
    _validate_mvp_runtime_contract(lowered_ir, target)
    return _write_constants(artifact_dir, lowered_ir, spec.constants, target=target), reports


def _build_artifact_from_lowered_ir(
    spec: ModelSpec,
    target: Target,
    *,
    artifact_dir: Path,
    generated_src_dir: Path,
    lowered_ir: Mapping[str, Any],
    reports: Sequence[Any],
    backend: Any,
    execution_plan_payload: Mapping[str, Any] | None,
    constant_load_policy: str,
) -> Artifact:
    debug_dir = artifact_dir / "debug"
    lowered_ir = _strip_compile_only_metadata(lowered_ir)

    execution_plan_config = (
        _execution_plan_compile_config(execution_plan_payload)
        if execution_plan_payload is not None
        else None
    )
    compile_config = {
        "target": target.to_json(),
        "constant_load_policy": constant_load_policy,
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
    if execution_plan_config is not None:
        compile_config["execution_plan"] = execution_plan_config

    write_json(artifact_dir / "graph.dinoir.json", lowered_ir)
    write_json(artifact_dir / "metadata.json", _runtime_metadata(lowered_ir))
    encoded_constants_manifest = _encoded_constants_manifest(lowered_ir)
    if encoded_constants_manifest is not None:
        write_json(artifact_dir / "encoded_constants.json", encoded_constants_manifest)
    else:
        (artifact_dir / "encoded_constants.json").unlink(missing_ok=True)
    kernel_manifest = build_kernel_manifest(lowered_ir, target.to_json())
    _validate_gguf_runtime_dequant_admission(lowered_ir, target, kernel_manifest)
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
    codegen_plan_payload = codegen_plan.to_json()
    wrapper_sources = _materialize_wrapper_debug_sources(
        artifact_dir=artifact_dir,
        generated_src_dir=generated_src_dir,
        target=codegen_plan.target,
        wrapper_stages=codegen_plan.wrapper_stages,
    )
    if wrapper_sources:
        codegen_plan_payload["wrapper_manifest"] = "debug/generated_src/conv_wrapper_source_manifest.json"
        codegen_plan_payload["wrapper_sources"] = wrapper_sources
    write_json(artifact_dir / "kernel_codegen_plan.json", codegen_plan_payload)

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
    if encoded_constants_manifest is not None:
        files["encoded_constants"] = "encoded_constants.json"

    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "runtime_abi_version": RUNTIME_ABI_VERSION,
        "name": spec.name,
        "target": target.to_json(),
        "constant_load_policy": constant_load_policy,
        "files": files,
        "graph_hash": graph_hash(lowered_ir),
    }
    if execution_plan_config is not None:
        manifest["execution_plan"] = execution_plan_config
    write_json(artifact_dir / "manifest.json", manifest)

    build_files = backend.resolve_build_function()(
        lowered_ir,
        target=target,
        artifact_dir=artifact_dir,
        generated_src_dir=generated_src_dir,
        kernel_manifest=kernel_manifest,
    )
    if build_files:
        manifest["files"].update(dict(build_files))
        write_json(artifact_dir / "manifest.json", manifest)

    return Artifact(artifact_dir)


def _strip_compile_only_metadata(ir: Mapping[str, Any]) -> dict[str, Any]:
    runtime_ir = dict(ir)
    metadata = runtime_ir.get("metadata")
    if not isinstance(metadata, Mapping):
        return runtime_ir
    runtime_metadata = dict(metadata)
    runtime_metadata.pop("profiling", None)
    runtime_ir["metadata"] = runtime_metadata
    return runtime_ir


def _load_execution_plan(execution_plan: str | Path | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if execution_plan is None:
        return None
    if isinstance(execution_plan, Mapping):
        return dict(execution_plan)
    return read_json(Path(execution_plan))


def _materialize_wrapper_debug_sources(
    *,
    artifact_dir: Path,
    generated_src_dir: Path,
    target: Mapping[str, Any],
    wrapper_stages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    stage_groups = _group_wrapper_stages(wrapper_stages)
    if not stage_groups:
        (generated_src_dir / "conv_wrapper_source_manifest.json").unlink(missing_ok=True)
        return []
    sources: list[dict[str, Any]] = []
    for stages in stage_groups:
        first_stage = stages[0]
        op_name = str(first_stage.get("op") or "conv2d_bias")
        node_id = None if first_stage.get("node_id") is None else str(first_stage.get("node_id"))
        source_key = canonical_json(
            {
                "kind": "cutlass_conv_wrapper",
                "op": op_name,
                "node_id": node_id,
                "stages": [dict(stage) for stage in stages],
            }
        )
        source_hash = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:16]
        file_name = f"{_wrapper_file_stem(node_id=node_id, op_name=op_name)}_{source_hash}.cu"
        emitted_source_path = Path("debug") / "generated_src" / "conv_wrappers" / op_name / file_name
        full_source_path = artifact_dir / emitted_source_path
        full_source_path.parent.mkdir(parents=True, exist_ok=True)
        full_source_path.write_text(
            render_conv_wrapper_source(stages, op_name=op_name, node_id=node_id),
            encoding="utf-8",
        )
        sources.append(
            {
                "source_kind": "cutlass_conv_wrapper",
                "kernel_library": str(first_stage.get("kernel_library") or "cutlass_conv"),
                "op": op_name,
                "node_id": node_id,
                "stage_names": [str(stage.get("stage_name") or "") for stage in stages],
                "stage_count": len(stages),
                "blocked_reason": str(
                    first_stage.get("blocked_reason") or "cutlass_conv_runtime_launcher_not_implemented"
                ),
                "source_key": source_key,
                "source_hash": source_hash,
                "emitted_source_path": emitted_source_path.as_posix(),
            }
        )
    manifest = {
        "schema_version": 1,
        "kind": "dinoml.conv_wrapper_source_manifest",
        "target": dict(target),
        "sources": sources,
    }
    (generated_src_dir / "conv_wrapper_source_manifest.json").write_text(
        canonical_json(manifest),
        encoding="utf-8",
    )
    return sources


def _group_wrapper_stages(
    wrapper_stages: Sequence[Mapping[str, Any]],
) -> list[list[Mapping[str, Any]]]:
    groups: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    current_key: tuple[str, str | None, str] | None = None
    for stage in wrapper_stages:
        key = (
            str(stage.get("op") or ""),
            None if stage.get("node_id") is None else str(stage.get("node_id")),
            str(stage.get("kernel_library") or ""),
        )
        if current and key != current_key:
            groups.append(current)
            current = []
        current.append(dict(stage))
        current_key = key
    if current:
        groups.append(current)
    return groups


def _wrapper_file_stem(*, node_id: str | None, op_name: str) -> str:
    base = node_id or op_name
    pieces = [char if char.isalnum() else "_" for char in str(base)]
    stem = "".join(pieces).strip("_")
    return stem or "cutlass_conv_wrapper"


def _validate_constant_load_policy(policy: str) -> str:
    normalized = str(policy)
    if normalized not in {"eager", "deferred"}:
        raise ValueError(f"Unsupported constant_load_policy {policy!r}; expected 'eager' or 'deferred'")
    return normalized


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
    _validate_execution_plan_key(execution_plan)
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


def _validate_execution_plan_key(execution_plan: Mapping[str, Any]) -> None:
    plan_key = execution_plan.get("execution_plan_key")
    if plan_key is None:
        return
    if not isinstance(plan_key, str) or not plan_key:
        raise ValueError("Execution plan key must be a non-empty string")
    expected = hashlib.sha256(
        canonical_json({key: value for key, value in execution_plan.items() if key != "execution_plan_key"}).encode(
            "utf-8"
        )
    ).hexdigest()
    if plan_key != expected:
        raise ValueError(f"Execution plan key does not match payload ({plan_key} != {expected})")


def _execution_plan_compile_config(execution_plan: Mapping[str, Any]) -> dict[str, Any]:
    summary = dict(execution_plan.get("summary", {})) if isinstance(execution_plan.get("summary"), Mapping) else {}
    return {
        "schema_version": execution_plan.get("schema_version"),
        "execution_plan_key": execution_plan.get("execution_plan_key"),
        "kernel_manifest_cache_key": execution_plan.get("kernel_manifest_cache_key"),
        "selection_policy": execution_plan.get("selection_policy"),
        "selection_confidence_policy": execution_plan.get("selection_confidence_policy"),
        "static_selection_policy": execution_plan.get("static_selection_policy"),
        "summary": summary,
    }


def _runtime_metadata(ir: Dict) -> Dict:
    return {
        "runtime_abi_version": RUNTIME_ABI_VERSION,
        "name": ir["name"],
        "inputs": ir["inputs"],
        "states": ir.get("states", []),
        "outputs": ir["outputs"],
        "constants": ir["constants"],
        "memory_plan": ir.get("metadata", {}).get("memory_plan", {}),
        "output_shape_reports": ir.get("metadata", {}).get("output_shape_reports", {}),
    }


def _requires_kernel_library(kernel_manifest: Dict, library: str) -> bool:
    return any(item.get("kernel_library") == library for item in kernel_manifest.get("required_kernels", []))


def _write_constants(artifact_dir: Path, ir: Dict, constants: Mapping[str, Any], *, target: Target) -> Dict:
    prepacked_cutlass_conv_weights = _cuda_cutlass_conv_weight_constants(ir) if target.name == "cuda" else set()
    prepacked_ck_conv1d_weights = _rocm_ck_conv1d_weight_constants(ir) if target.name == "rocm" else set()
    offset = 0
    constant_infos = []
    with (artifact_dir / "constants.bin").open("wb") as handle:
        for constant in ir["constants"]:
            name = constant["name"]
            if name not in constants:
                raise ValueError(f"Missing constant value: {name}")
            expected_shape = tuple(int(dim) for dim in constant["shape"])
            value = constants[name]
            if _writes_encoded_gguf_runtime_dequant_constant(value):
                materialized = materialize_gguf_encoded_constant(value, constant["dtype"], expected_shape)
                data = materialized.array.tobytes(order="C")
            else:
                materialized = materialize_constant_value(value, constant["dtype"], expected_shape)
                array = materialized.array
                if array.shape != expected_shape:
                    raise ValueError(f"Constant {name} has shape {array.shape}, expected {expected_shape}")
                expected_dtype = dtype_numpy(str(constant["dtype"]))
                if array.dtype != expected_dtype:
                    raise ValueError(f"Constant {name} has storage dtype {array.dtype}, expected {expected_dtype}")
            prepack_storage = None
            if constant["tensor"] in prepacked_cutlass_conv_weights:
                if array.ndim == 4:
                    array = np.ascontiguousarray(np.transpose(array, (0, 2, 3, 1)))
                    logical_layout = "oihw"
                    storage_layout = "ohwi"
                elif array.ndim == 3:
                    array = np.ascontiguousarray(np.transpose(array, (0, 2, 1)))
                    logical_layout = "oiw"
                    storage_layout = "owi"
                else:
                    raise ValueError(f"CUTLASS Conv weight constant {name} must be rank-3 OIW or rank-4 OIHW before packing")
                prepack_storage = {
                    **(dict(constant.get("storage", {})) if isinstance(constant.get("storage"), Mapping) else {}),
                    "kind": "cutlass_conv_weight",
                    "logical_layout": logical_layout,
                    "storage_layout": storage_layout,
                }
            elif constant["tensor"] in prepacked_ck_conv1d_weights:
                if array.ndim != 3:
                    raise ValueError(f"CK Conv1d weight constant {name} must be rank-3 OIW before packing")
                array = np.ascontiguousarray(np.transpose(array, (0, 2, 1)))
                prepack_storage = {
                    **(dict(constant.get("storage", {})) if isinstance(constant.get("storage"), Mapping) else {}),
                    "kind": "ck_conv1d_weight",
                    "logical_layout": "oiw",
                    "storage_layout": "kxc",
                }
            data = array.tobytes(order="C")
            constant = dict(constant)
            constant["offset"] = offset
            constant["nbytes"] = len(data)
            if materialized.storage is not None:
                storage = dict(materialized.storage)
                if prepack_storage is not None:
                    storage = {**storage, **prepack_storage}
                constant["storage"] = storage
            elif prepack_storage is not None:
                constant["storage"] = prepack_storage
            else:
                constant.pop("storage", None)
            constant_infos.append(constant)
            handle.write(data)
            offset += len(data)
    ir = dict(ir)
    ir["constants"] = constant_infos
    ir.setdefault("metadata", {})["constants_nbytes"] = offset
    return ir


def _cuda_cutlass_conv_weight_constants(ir: Mapping[str, Any]) -> set[str]:
    constants = {str(item["tensor"]) for item in ir.get("constants", [])}
    result: set[str] = set()
    for node in ir.get("nodes", []):
        if str(node.get("op", "")) not in {
            "conv1d_bias",
            "conv1d_bias_relu",
            "conv1d_bias_add",
            "conv1d_bias_add_relu",
            "conv2d_bias",
            "conv2d_bias_relu",
            "conv2d_bias_add",
            "conv2d_bias_add_relu",
        }:
            continue
        inputs = node.get("inputs", ())
        if not isinstance(inputs, Sequence) or len(inputs) < 2:
            continue
        weight = str(inputs[1])
        if weight in constants:
            result.add(weight)
    return result


def _rocm_ck_conv1d_weight_constants(ir: Mapping[str, Any]) -> set[str]:
    constants = {str(item["tensor"]) for item in ir.get("constants", [])}
    result: set[str] = set()
    for node in ir.get("nodes", []):
        if str(node.get("op", "")) not in {"conv1d_bias", "conv1d_bias_relu", "conv1d_bias_add", "conv1d_bias_add_relu"}:
            continue
        inputs = node.get("inputs", ())
        if not isinstance(inputs, Sequence) or len(inputs) < 2:
            continue
        weight = str(inputs[1])
        if weight in constants:
            result.add(weight)
    return result


def _writes_encoded_gguf_runtime_dequant_constant(value: Any) -> bool:
    return (
        isinstance(value, GGUFConstant)
        and value.materialization == GGUF_MATERIALIZATION_DEQUANTIZE_ON_GPU_BEFORE_LAUNCH
        and value.residency == GGUF_RESIDENCY_MANUAL_RUNTIME_LOAD
    )


def _shape_nbytes(shape: Sequence[int], dtype: str) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel * dtype_nbytes(dtype)


def _encoded_constants_manifest(ir: Mapping[str, Any]) -> dict[str, Any] | None:
    encoded_constants = []
    total_logical_nbytes = 0
    total_encoded_nbytes = 0
    for constant in ir.get("constants", []):
        storage = constant.get("storage")
        if not isinstance(storage, Mapping):
            continue
        if storage.get("kind") != "gguf":
            continue
        materialization = str(storage.get("materialization", GGUF_MATERIALIZATION_DEQUANTIZE_FULL_BEFORE_LAUNCH))
        residency = str(storage.get("residency", GGUF_RESIDENCY_EAGER_DENSE_DEVICE))
        policy_status = gguf_constant_policy_status(materialization, residency)
        if materialization == GGUF_MATERIALIZATION_DEQUANTIZE_ON_GPU_BEFORE_LAUNCH:
            logical_nbytes = _shape_nbytes(constant.get("shape", []), str(constant["dtype"]))
        else:
            logical_nbytes = int(constant.get("nbytes", 0) or 0)
        encoded_nbytes = int(storage.get("encoded_nbytes", 0) or 0)
        total_logical_nbytes += logical_nbytes
        total_encoded_nbytes += encoded_nbytes
        encoded_constants.append(
            {
                "name": constant["name"],
                "tensor": constant.get("tensor", constant["name"]),
                "dtype": constant["dtype"],
                "shape": list(constant.get("shape", [])),
                "logical_nbytes": logical_nbytes,
                "storage": dict(storage),
                "policy": {
                    "materialization": materialization,
                    "materialization_status": policy_status["materialization"],
                    "residency": residency,
                    "residency_status": policy_status["residency"],
                },
            }
        )
    if not encoded_constants:
        return None
    return {
        "schema_version": 1,
        "kind": "dinoml.encoded_constants",
        "constants": encoded_constants,
        "summary": {
            "constant_count": len(encoded_constants),
            "logical_nbytes": total_logical_nbytes,
            "encoded_nbytes": total_encoded_nbytes,
            "runtime_supported_count": sum(
                1
                for item in encoded_constants
                if item["policy"]["materialization_status"] == "runtime_supported"
                and item["policy"]["residency_status"] == "runtime_supported"
            ),
        },
    }


def _validate_gguf_runtime_dequant_admission(
    ir: Mapping[str, Any],
    target: Target,
    kernel_manifest: Mapping[str, Any],
) -> None:
    tensor_map = {str(tensor["name"]): tensor for tensor in ir.get("tensors", [])}
    runtime_dequant_tensors = {
        str(constant["tensor"]): {
            "name": str(constant["name"]),
            "residency": str(constant["storage"].get("residency", GGUF_RESIDENCY_EAGER_DENSE_DEVICE)),
        }
        for constant in ir.get("constants", [])
        if isinstance(constant.get("storage"), Mapping)
        and str(constant["storage"].get("kind", "")) == "gguf"
        and str(constant["storage"].get("materialization", "")) == GGUF_MATERIALIZATION_DEQUANTIZE_ON_GPU_BEFORE_LAUNCH
    }
    if not runtime_dequant_tensors:
        return
    if target.name != "cuda":
        names = ", ".join(sorted(str(info["name"]) for info in runtime_dequant_tensors.values()))
        raise NotImplementedError(
            "GGUF materialization='dequantize_on_gpu_before_launch' is only supported for CUDA gemm_rrr/gemm_rcr "
            "or gemm_rrr_bias/gemm_rcr_bias RHS constants; "
            f"unsupported target {target.name!r} for constant(s): {names}"
        )
    lowered_constants = {
        str(plan.get("constant"))
        for item in kernel_manifest.get("required_kernels", [])
        if isinstance(item, Mapping)
        for plan in [item.get("gguf_runtime_dequant")]
        if isinstance(plan, Mapping) and str(plan.get("status", "")) == "lowered_runtime_dequant_scratch"
    }
    unsupported_uses: dict[str, list[str]] = {}
    for node in ir.get("nodes", []):
        inputs = [str(name) for name in node.get("inputs", [])]
        output_name = str(node.get("outputs", [""])[0])
        output_dtype = str(tensor_map.get(output_name, {}).get("dtype", ""))
        for input_index, tensor_name in enumerate(inputs):
            constant_info = runtime_dequant_tensors.get(tensor_name)
            if constant_info is None:
                continue
            constant_name = str(constant_info["name"])
            residency = str(constant_info["residency"])
            if residency != GGUF_RESIDENCY_MANUAL_RUNTIME_LOAD:
                unsupported_uses.setdefault(constant_name, []).append(f"unsupported_residency:{residency}")
            supported_use = (
                input_index == 1
                and str(node.get("op", "")) in {"gemm_rrr", "gemm_rcr", "gemm_rrr_bias", "gemm_rcr_bias"}
                and output_dtype in {"float32", "float16"}
                and residency == GGUF_RESIDENCY_MANUAL_RUNTIME_LOAD
                and constant_name in lowered_constants
            )
            if not supported_use:
                unsupported_uses.setdefault(constant_name, []).append(f"{node.get('op')}[input {input_index}]")
    for constant_info in runtime_dequant_tensors.values():
        constant_name = str(constant_info["name"])
        if constant_name not in lowered_constants:
            unsupported_uses.setdefault(constant_name, []).append("no_supported_lowered_use")
    if not unsupported_uses:
        return
    details = "; ".join(
        f"{name}: {', '.join(uses)}"
        for name, uses in sorted(unsupported_uses.items())
    )
    raise NotImplementedError(
        "GGUF materialization='dequantize_on_gpu_before_launch' is only supported as the CUDA "
        "gemm_rrr/gemm_rcr or gemm_rrr_bias/gemm_rcr_bias RHS for float32/float16 output "
        "with residency='manual_runtime_load'; unsupported uses: "
        f"{details}"
    )


def _validate_mvp_runtime_contract(ir: Dict, target: Target) -> None:
    tensor_map = {tensor["name"]: tensor for tensor in ir["tensors"]}
    index_tensors = {
        node["inputs"][1]
        for node in ir["nodes"]
        if node.get("op") in {"gather", "batch_gather", "embedding", "runtime_index_select"}
        and len(node.get("inputs", [])) == 2
    }
    index_tensors.update(
        node["inputs"][1]
        for node in ir["nodes"]
        if node.get("op") == "index_add" and len(node.get("inputs", [])) == 3
    )
    index_tensors.update(
        node["inputs"][0]
        for node in ir["nodes"]
        if node.get("op") in {"glm_ocr_stitch_image_features", "qwen2_5_vl_stitch_image_features"}
        and len(node.get("inputs", [])) == 3
    )
    argmax_input_tensors = {
        node["inputs"][0]
        for node in ir["nodes"]
        if node.get("op") == "argmax" and len(node.get("inputs", [])) == 1
    }
    argmax_output_tensors = {
        node["outputs"][0]
        for node in ir["nodes"]
        if node.get("op") == "argmax" and len(node.get("outputs", [])) == 1
    }
    topk_index_output_tensors = {
        node["outputs"][0]
        for node in ir["nodes"]
        if node.get("op") == "topk_indices" and len(node.get("outputs", [])) == 1
    }
    nms_int64_output_tensors = {
        node["outputs"][0]
        for node in ir["nodes"]
        if node.get("op") == "batched_nms" and len(node.get("outputs", [])) == 1
    }
    nms_int64_output_tensors.update(
        output_name
        for node in ir["nodes"]
        if node.get("op") == "efficient_nms" and len(node.get("outputs", [])) == 4
        for output_name in (node["outputs"][0], node["outputs"][3])
    )
    rotary_allegro_grid_output_tensors = {
        output_name
        for node in ir["nodes"]
        if node.get("op") == "get_3d_rotary_pos_embed_allegro" and len(node.get("outputs", [])) == 9
        for output_name in node["outputs"][6:]
    }
    flash_attention_static_kv_cache_seqlens_tensors = {
        node["inputs"][5]
        for node in ir["nodes"]
        if node.get("op") in {"flash_attention_static_kv_cache", "flash_attention_static_kv_cache_bias"}
        and len(node.get("inputs", [])) >= 6
    }
    flash_attention_varlen_cu_seqlens_tensors = {
        node["inputs"][3]
        for node in ir["nodes"]
        if node.get("op") == "flash_attention_varlen" and len(node.get("inputs", [])) == 4
    }
    fused_integer_eq_tensors = {
        tensor_name
        for node in ir["nodes"]
        if _node_is_integer_eq_fused_elementwise(node, tensor_map)
        for tensor_name in node.get("inputs", [])
        if str(tensor_map[tensor_name]["dtype"]) in {"int32", "int64"}
    }
    for node in ir["nodes"]:
        op_def = get_op_def(str(node["op"]))
        if target.name not in op_def.backend_kernels:
            raise NotImplementedError(f"{target.name} backend does not support op {op_def.name}")
        if node.get("op") == "argmax":
            input_dtype = str(tensor_map[node["inputs"][0]]["dtype"])
            output_dtype = str(tensor_map[node["outputs"][0]]["dtype"])
            if input_dtype not in op_def.allowed_dtypes:
                raise NotImplementedError(
                    f"Op {op_def.name} supports input dtypes {list(op_def.allowed_dtypes)}; "
                    f"unsupported compiled dtypes: {[input_dtype]}"
                )
            if output_dtype != "int64":
                raise NotImplementedError(f"Op argmax output dtype {output_dtype} must be int64")
            continue
        if node.get("op") == "batched_nms":
            input_dtype = str(tensor_map[node["inputs"][0]]["dtype"])
            output_dtype = str(tensor_map[node["outputs"][0]]["dtype"])
            if input_dtype not in op_def.allowed_dtypes:
                raise NotImplementedError(
                    f"Op {op_def.name} supports input dtypes {list(op_def.allowed_dtypes)}; "
                    f"unsupported compiled dtypes: {[input_dtype]}"
                )
            if output_dtype != "int64":
                raise NotImplementedError(f"Op batched_nms output dtype {output_dtype} must be int64")
            continue
        if node.get("op") == "efficient_nms":
            box_dtype = str(tensor_map[node["inputs"][0]]["dtype"])
            score_dtype = str(tensor_map[node["inputs"][1]]["dtype"])
            output_dtypes = [str(tensor_map[name]["dtype"]) for name in node["outputs"]]
            if box_dtype not in op_def.allowed_dtypes:
                raise NotImplementedError(
                    f"Op {op_def.name} supports input dtypes {list(op_def.allowed_dtypes)}; "
                    f"unsupported compiled dtypes: {[box_dtype]}"
                )
            if score_dtype != box_dtype:
                raise NotImplementedError(
                    f"Op efficient_nms scores dtype {score_dtype} must match boxes dtype {box_dtype}"
                )
            expected_output_dtypes = ["int64", box_dtype, box_dtype, "int64"]
            if output_dtypes != expected_output_dtypes:
                raise NotImplementedError(
                    f"Op efficient_nms output dtypes {output_dtypes} must be {expected_output_dtypes}"
                )
            continue
        if node.get("op") in {"gather", "runtime_index_select"}:
            op_name = str(node["op"])
            data_dtype = str(tensor_map[node["inputs"][0]]["dtype"])
            index_dtype = str(tensor_map[node["inputs"][1]]["dtype"])
            output_dtype = str(tensor_map[node["outputs"][0]]["dtype"])
            if data_dtype not in op_def.allowed_dtypes:
                raise NotImplementedError(
                    f"Op {op_def.name} supports dtypes {list(op_def.allowed_dtypes)}; "
                    f"unsupported compiled dtypes: {[data_dtype]}"
                )
            if index_dtype not in {"int64", "int32"}:
                raise NotImplementedError(
                    f"Op {op_name} index supports dtypes ['int64', 'int32']; "
                    f"unsupported compiled dtypes: {[index_dtype]}"
                )
            if output_dtype != data_dtype:
                raise NotImplementedError(f"Op {op_name} output dtype {output_dtype} must match input dtype {data_dtype}")
            continue
        if node.get("op") == "index_add":
            data_dtype = str(tensor_map[node["inputs"][0]]["dtype"])
            index_dtype = str(tensor_map[node["inputs"][1]]["dtype"])
            source_dtype = str(tensor_map[node["inputs"][2]]["dtype"])
            output_dtype = str(tensor_map[node["outputs"][0]]["dtype"])
            if data_dtype not in INDEX_ADD_DTYPES:
                raise NotImplementedError(
                    f"Op index_add supports dtypes {list(INDEX_ADD_DTYPES)}; "
                    f"unsupported compiled dtypes: {[data_dtype]}"
                )
            if index_dtype not in {"int64", "int32"}:
                raise NotImplementedError(
                    "Op index_add index supports dtypes ['int64', 'int32']; "
                    f"unsupported compiled dtypes: {[index_dtype]}"
                )
            if source_dtype != data_dtype:
                raise NotImplementedError(
                    f"Op index_add source dtype {source_dtype} must match input dtype {data_dtype}"
                )
            if output_dtype != data_dtype:
                raise NotImplementedError(f"Op index_add output dtype {output_dtype} must match input dtype {data_dtype}")
            continue
        if node.get("op") == "masked_select":
            data_dtype = str(tensor_map[node["inputs"][0]]["dtype"])
            mask_dtype = str(tensor_map[node["inputs"][1]]["dtype"])
            output_dtype = str(tensor_map[node["outputs"][0]]["dtype"])
            if data_dtype not in op_def.allowed_dtypes:
                raise NotImplementedError(
                    f"Op {op_def.name} supports dtypes {list(op_def.allowed_dtypes)}; "
                    f"unsupported compiled dtypes: {[data_dtype]}"
                )
            if mask_dtype != "bool":
                raise NotImplementedError(
                    "Op masked_select mask supports dtype ['bool']; "
                    f"unsupported compiled dtypes: {[mask_dtype]}"
                )
            if output_dtype != data_dtype:
                raise NotImplementedError(
                    f"Op masked_select output dtype {output_dtype} must match input dtype {data_dtype}"
                )
            continue
        if node.get("op") == "batch_gather":
            data_dtype = str(tensor_map[node["inputs"][0]]["dtype"])
            index_dtype = str(tensor_map[node["inputs"][1]]["dtype"])
            output_dtype = str(tensor_map[node["outputs"][0]]["dtype"])
            if data_dtype not in op_def.allowed_dtypes:
                raise NotImplementedError(
                    f"Op {op_def.name} supports dtypes {list(op_def.allowed_dtypes)}; "
                    f"unsupported compiled dtypes: {[data_dtype]}"
                )
            if index_dtype not in {"int64", "int32"}:
                raise NotImplementedError(
                    "Op batch_gather indices support dtypes ['int64', 'int32']; "
                    f"unsupported compiled dtypes: {[index_dtype]}"
                )
            if output_dtype != data_dtype:
                raise NotImplementedError(
                    f"Op batch_gather output dtype {output_dtype} must match input dtype {data_dtype}"
                )
            continue
        if node.get("op") == "embedding":
            table_dtype = str(tensor_map[node["inputs"][0]]["dtype"])
            index_dtype = str(tensor_map[node["inputs"][1]]["dtype"])
            output_dtype = str(tensor_map[node["outputs"][0]]["dtype"])
            if table_dtype not in op_def.allowed_dtypes:
                raise NotImplementedError(
                    f"Op {op_def.name} supports dtypes {list(op_def.allowed_dtypes)}; "
                    f"unsupported compiled dtypes: {[table_dtype]}"
                )
            if index_dtype not in {"int64", "int32"}:
                raise NotImplementedError(
                    "Op embedding indices support dtypes ['int64', 'int32']; "
                    f"unsupported compiled dtypes: {[index_dtype]}"
                )
            if output_dtype != table_dtype:
                raise NotImplementedError(
                    f"Op embedding output dtype {output_dtype} must match table dtype {table_dtype}"
                )
            continue
        if node.get("op") == "glm_ocr_stitch_image_features":
            index_dtype = str(tensor_map[node["inputs"][0]]["dtype"])
            data_dtype = str(tensor_map[node["inputs"][1]]["dtype"])
            image_features_dtype = str(tensor_map[node["inputs"][2]]["dtype"])
            output_dtype = str(tensor_map[node["outputs"][0]]["dtype"])
            if data_dtype not in op_def.allowed_dtypes:
                raise NotImplementedError(
                    f"Op {op_def.name} supports dtypes {list(op_def.allowed_dtypes)}; "
                    f"unsupported compiled dtypes: {[data_dtype]}"
                )
            if index_dtype not in {"int64", "int32"}:
                raise NotImplementedError(
                    "Op glm_ocr_stitch_image_features input_ids support dtypes ['int64', 'int32']; "
                    f"unsupported compiled dtypes: {[index_dtype]}"
                )
            if image_features_dtype != data_dtype:
                raise NotImplementedError(
                    "Op glm_ocr_stitch_image_features image_features dtype must match "
                    f"inputs_embeds dtype {data_dtype}, got {image_features_dtype}"
                )
            if output_dtype != data_dtype:
                raise NotImplementedError(
                    f"Op glm_ocr_stitch_image_features output dtype {output_dtype} must match input dtype {data_dtype}"
                )
            continue
        if node.get("op") == "qwen2_5_vl_stitch_image_features":
            index_dtype = str(tensor_map[node["inputs"][0]]["dtype"])
            data_dtype = str(tensor_map[node["inputs"][1]]["dtype"])
            image_features_dtype = str(tensor_map[node["inputs"][2]]["dtype"])
            output_dtype = str(tensor_map[node["outputs"][0]]["dtype"])
            if data_dtype not in op_def.allowed_dtypes:
                raise NotImplementedError(
                    f"Op {op_def.name} supports dtypes {list(op_def.allowed_dtypes)}; "
                    f"unsupported compiled dtypes: {[data_dtype]}"
                )
            if index_dtype not in {"int64", "int32"}:
                raise NotImplementedError(
                    "Op qwen2_5_vl_stitch_image_features input_ids support dtypes ['int64', 'int32']; "
                    f"unsupported compiled dtypes: {[index_dtype]}"
                )
            if image_features_dtype != data_dtype:
                raise NotImplementedError(
                    "Op qwen2_5_vl_stitch_image_features image_features dtype must match "
                    f"inputs_embeds dtype {data_dtype}, got {image_features_dtype}"
                )
            if output_dtype != data_dtype:
                raise NotImplementedError(
                    f"Op qwen2_5_vl_stitch_image_features output dtype {output_dtype} must match input dtype {data_dtype}"
                )
            continue
        if node.get("op") in {"topk_values", "topk_indices"}:
            input_dtype = str(tensor_map[node["inputs"][0]]["dtype"])
            output_dtype = str(tensor_map[node["outputs"][0]]["dtype"])
            if input_dtype not in op_def.allowed_dtypes:
                raise NotImplementedError(
                    f"Op {op_def.name} supports input dtypes {list(op_def.allowed_dtypes)}; "
                    f"unsupported compiled dtypes: {[input_dtype]}"
                )
            expected_output_dtype = input_dtype if node.get("op") == "topk_values" else "int64"
            if output_dtype != expected_output_dtype:
                raise NotImplementedError(
                    f"Op {op_def.name} output dtype {output_dtype} must be {expected_output_dtype}"
                )
            continue
        if node.get("op") in {"get_1d_rotary_pos_embed_cos", "get_1d_rotary_pos_embed_sin"}:
            input_dtype = None if not node.get("inputs") else str(tensor_map[node["inputs"][0]]["dtype"])
            output_dtype = str(tensor_map[node["outputs"][0]]["dtype"])
            if input_dtype is not None and input_dtype != "float32":
                raise NotImplementedError(
                    f"Op {op_def.name} requires float32 pos input; unsupported compiled dtypes: {[input_dtype]}"
                )
            if output_dtype not in op_def.allowed_dtypes:
                raise NotImplementedError(
                    f"Op {op_def.name} supports output dtypes {list(op_def.allowed_dtypes)}; "
                    f"unsupported compiled dtypes: {[output_dtype]}"
                )
            continue
        if node.get("op") in {
            "get_2d_rotary_pos_embed",
            "get_2d_rotary_pos_embed_lumina",
            "get_3d_rotary_pos_embed",
            "get_3d_rotary_pos_embed_allegro",
        }:
            output_dtypes = [str(tensor_map[name]["dtype"]) for name in node["outputs"]]
            float_dtypes = output_dtypes if node.get("op") != "get_3d_rotary_pos_embed_allegro" else output_dtypes[:6]
            unsupported_float_dtypes = [dtype for dtype in float_dtypes if dtype not in {"float16", "float32", "bfloat16"}]
            if unsupported_float_dtypes:
                raise NotImplementedError(
                    f"Op {op_def.name} supports float outputs ['float16', 'float32', 'bfloat16']; "
                    f"unsupported compiled dtypes: {unsupported_float_dtypes}"
                )
            if node.get("op") == "get_3d_rotary_pos_embed_allegro":
                grid_dtypes = output_dtypes[6:]
                if any(dtype != "int64" for dtype in grid_dtypes):
                    raise NotImplementedError(
                        "Op get_3d_rotary_pos_embed_allegro grid outputs must use int64; "
                        f"unsupported compiled dtypes: {grid_dtypes}"
                    )
            continue
        if node.get("op") in {"flash_attention_static_kv_cache", "flash_attention_static_kv_cache_bias"}:
            op_name = str(node["op"])
            data_input_names = list(node["inputs"][:5])
            if op_name == "flash_attention_static_kv_cache_bias":
                data_input_names.append(node["inputs"][6])
            data_dtypes = sorted({str(tensor_map[name]["dtype"]) for name in [*data_input_names, *node["outputs"]]})
            unsupported_data_dtypes = [dtype for dtype in data_dtypes if dtype not in op_def.allowed_dtypes]
            if unsupported_data_dtypes:
                raise NotImplementedError(
                    f"Op {op_def.name} supports data dtypes {list(op_def.allowed_dtypes)}; "
                    f"unsupported compiled dtypes: {unsupported_data_dtypes}"
                )
            cache_seqlens_dtype = str(tensor_map[node["inputs"][5]]["dtype"])
            if cache_seqlens_dtype != "int32":
                raise NotImplementedError(
                    f"Op {op_name} cache_seqlens supports dtype ['int32']; "
                    f"unsupported compiled dtypes: {[cache_seqlens_dtype]}"
                )
            continue
        if node.get("op") == "flash_attention_varlen":
            data_input_names = list(node["inputs"][:3])
            data_dtypes = sorted({str(tensor_map[name]["dtype"]) for name in [*data_input_names, *node["outputs"]]})
            unsupported_data_dtypes = [dtype for dtype in data_dtypes if dtype not in op_def.allowed_dtypes]
            if unsupported_data_dtypes:
                raise NotImplementedError(
                    f"Op {op_def.name} supports data dtypes {list(op_def.allowed_dtypes)}; "
                    f"unsupported compiled dtypes: {unsupported_data_dtypes}"
                )
            cu_seqlens_dtype = str(tensor_map[node["inputs"][3]]["dtype"])
            if cu_seqlens_dtype != "int32":
                raise NotImplementedError(
                    "Op flash_attention_varlen cu_seqlens supports dtype ['int32']; "
                    f"unsupported compiled dtypes: {[cu_seqlens_dtype]}"
                )
            continue
        node_tensor_names = [*node.get("inputs", []), *node.get("outputs", [])]
        node_dtypes = sorted({tensor_map[name]["dtype"] for name in node_tensor_names if name in tensor_map})
        unsupported_node_dtypes = [dtype for dtype in node_dtypes if dtype not in op_def.allowed_dtypes]
        if unsupported_node_dtypes:
            raise NotImplementedError(
                f"Op {op_def.name} supports dtypes {list(op_def.allowed_dtypes)}; "
                f"unsupported compiled dtypes: {unsupported_node_dtypes}"
            )
    supported = get_backend_spec(target.name).supported_dtypes
    unsupported = sorted(
        {
            str(tensor["dtype"])
            for tensor in ir["tensors"]
            if str(tensor["dtype"]) not in supported
            and str(tensor["name"]) not in index_tensors
            and str(tensor["name"]) not in argmax_input_tensors
            and str(tensor["name"]) not in argmax_output_tensors
            and str(tensor["name"]) not in topk_index_output_tensors
            and str(tensor["name"]) not in nms_int64_output_tensors
            and str(tensor["name"]) not in rotary_allegro_grid_output_tensors
            and str(tensor["name"]) not in flash_attention_static_kv_cache_seqlens_tensors
            and str(tensor["name"]) not in flash_attention_varlen_cu_seqlens_tensors
            and str(tensor["name"]) not in fused_integer_eq_tensors
        }
    )
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


def _node_is_integer_eq_fused_elementwise(
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> bool:
    if node.get("op") != "fused_elementwise":
        return False
    sub_ops = node.get("attrs", {}).get("sub_ops", [])
    if not isinstance(sub_ops, list) or not sub_ops:
        return False
    if any(str(sub_op.get("op")) != "eq" for sub_op in sub_ops):
        return False
    integer_inputs = [
        str(tensor_map[name]["dtype"])
        for name in node.get("inputs", [])
        if name in tensor_map and str(tensor_map[name]["dtype"]) in {"int32", "int64"}
    ]
    if not integer_inputs:
        return False
    return all(dtype in {"int32", "int64"} for dtype in integer_inputs)


def _validate_profile_shape_expressions(ir: Mapping[str, Any], target: Target) -> None:
    if target.name != "cuda":
        return
    tensor_map = {str(tensor["name"]): tensor for tensor in ir.get("tensors", [])}
    dynamic_dims = _profile_direct_dim_sources(ir)
    validate_symbolic_int_sources(items=ir.get("inputs", []), dynamic_dims=dynamic_dims, context="input")
    validate_symbolic_int_sources(items=ir.get("outputs", []), dynamic_dims=dynamic_dims, context="output")
    validate_symbolic_int_sources(items=ir.get("constants", []), dynamic_dims=dynamic_dims, context="constant")
    validate_symbolic_int_sources(items=tensor_map.values(), dynamic_dims=dynamic_dims, context="tensor")
    for view in ir.get("metadata", {}).get("memory_plan", {}).get("views", {}).get("views", []):
        validate_symbolic_int_sources(
            items=[
                {
                    "name": str(view.get("tensor", "<unknown>")),
                    "shape": view.get("shape", []),
                    "shape_spec": view.get("shape_spec", view.get("shape", [])),
                }
            ],
            dynamic_dims=dynamic_dims,
            context="view",
        )


def _profile_direct_dim_sources(ir: Mapping[str, Any]) -> dict[str, str]:
    sources: dict[str, str] = {}
    for section in ("inputs", "constants"):
        for item in ir.get(section, []):
            for axis, dim in enumerate(item.get("shape_spec", item.get("shape", []))):
                if isinstance(dim, Mapping) and dim.get("kind") == "dim":
                    sources.setdefault(str(dim["name"]), f"{section}.{item.get('name', item.get('tensor'))}[{axis}]")
    return sources
