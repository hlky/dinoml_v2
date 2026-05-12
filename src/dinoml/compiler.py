from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

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
from dinoml.lowering.shape_buffers import validate_symbolic_int_sources
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
    constant_load_policy: str = "eager",
) -> Artifact:
    constant_load_policy = _validate_constant_load_policy(constant_load_policy)
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
            repeats=profile_repeats,
            input_shapes=profile_input_shapes,
            refresh=profile_refresh,
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
    constant_load_policy: str,
) -> Artifact:
    backend = get_backend_spec(target.name)
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
    _validate_profile_shape_expressions(lowered_ir, target)
    initial_artifact = _build_artifact_from_lowered_ir(
        spec,
        target,
        artifact_dir=artifact_dir,
        generated_src_dir=generated_src_dir,
        lowered_ir=lowered_ir,
        reports=reports,
        backend=backend,
        execution_plan_payload=None,
        constant_load_policy=constant_load_policy,
    )
    profile_report = profile_artifact(
        initial_artifact.path,
        input_shapes=input_shapes,
        iterations=iterations,
        repeats=repeats,
        refresh=refresh,
    )
    execution_plan_summary = profile_report.get("execution_plan", {})
    if not isinstance(execution_plan_summary, Mapping) or not execution_plan_summary.get("path"):
        raise ValueError("Profiler did not produce an execution plan")
    if int(execution_plan_summary.get("selection_count", 0) or 0) == 0:
        write_json(initial_artifact.path / "debug" / "bootstrap_profile_report.json", dict(profile_report))
        return initial_artifact
    execution_plan_payload = read_json(Path(str(execution_plan_summary["path"])))
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
    return _write_constants(artifact_dir, lowered_ir, spec.constants), reports


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
    lowered_ir = dict(lowered_ir)

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
    if _requires_kernel_library(kernel_manifest, "cutlass_bmm"):
        files["cutlass_bmm_library"] = "lib/libdinoml_cutlass_bmm.so"
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
        "outputs": ir["outputs"],
        "constants": ir["constants"],
        "memory_plan": ir.get("metadata", {}).get("memory_plan", {}),
        "output_shape_reports": ir.get("metadata", {}).get("output_shape_reports", {}),
    }


def _requires_kernel_library(kernel_manifest: Dict, library: str) -> bool:
    return any(item.get("kernel_library") == library for item in kernel_manifest.get("required_kernels", []))


def _write_constants(artifact_dir: Path, ir: Dict, constants: Mapping[str, Any]) -> Dict:
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
                data = array.tobytes(order="C")
            constant = dict(constant)
            constant["offset"] = offset
            constant["nbytes"] = len(data)
            if materialized.storage is not None:
                constant["storage"] = materialized.storage
            else:
                constant.pop("storage", None)
            constant_infos.append(constant)
            handle.write(data)
            offset += len(data)
    ir = dict(ir)
    ir["constants"] = constant_infos
    ir.setdefault("metadata", {})["constants_nbytes"] = offset
    return ir


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
        str(constant["tensor"]): str(constant["name"])
        for constant in ir.get("constants", [])
        if isinstance(constant.get("storage"), Mapping)
        and str(constant["storage"].get("kind", "")) == "gguf"
        and str(constant["storage"].get("materialization", "")) == GGUF_MATERIALIZATION_DEQUANTIZE_ON_GPU_BEFORE_LAUNCH
    }
    if not runtime_dequant_tensors:
        return
    if target.name != "cuda":
        names = ", ".join(sorted(runtime_dequant_tensors.values()))
        raise NotImplementedError(
            "GGUF materialization='dequantize_on_gpu_before_launch' is only supported for CUDA gemm_rrr RHS constants; "
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
            constant_name = runtime_dequant_tensors.get(tensor_name)
            if constant_name is None:
                continue
            supported_use = (
                input_index == 1
                and str(node.get("op", "")) == "gemm_rrr"
                and output_dtype in {"float32", "float16"}
                and constant_name in lowered_constants
            )
            if not supported_use:
                unsupported_uses.setdefault(constant_name, []).append(f"{node.get('op')}[input {input_index}]")
    for constant_name in sorted(runtime_dequant_tensors.values()):
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
        "gemm_rrr RHS for float32/float16 output; unsupported uses: "
        f"{details}"
    )


def _validate_mvp_runtime_contract(ir: Dict, target: Target) -> None:
    tensor_map = {tensor["name"]: tensor for tensor in ir["tensors"]}
    gather_index_tensors = {
        node["inputs"][1]
        for node in ir["nodes"]
        if node.get("op") in {"gather", "batch_gather"} and len(node.get("inputs", [])) == 2
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
        if node.get("op") == "gather":
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
                    "Op gather index supports dtypes ['int64', 'int32']; "
                    f"unsupported compiled dtypes: {[index_dtype]}"
                )
            if output_dtype != data_dtype:
                raise NotImplementedError(f"Op gather output dtype {output_dtype} must match input dtype {data_dtype}")
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
            and str(tensor["name"]) not in gather_index_tensors
            and str(tensor["name"]) not in argmax_output_tensors
            and str(tensor["name"]) not in topk_index_output_tensors
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
