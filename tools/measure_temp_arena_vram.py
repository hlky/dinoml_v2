from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

import dinoml.runtime as runtime
from dinoml.gpu_memory_validation import (
    hip_device_synchronize,
    hip_mem_get_info,
    load_hip_runtime,
    matching_bucket_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure GPU free-memory deltas for DinoML temp arena policies. "
            "Scenarios are evaluated via a validation-input builder module."
        )
    )
    parser.add_argument("--artifact", required=True, help="Path to a built DinoML artifact directory")
    parser.add_argument(
        "--validation-module",
        required=True,
        help="Python module name or .py file exposing build_validation_inputs(**kwargs)",
    )
    parser.add_argument(
        "--common-kwarg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Keyword argument passed to every build_validation_inputs call",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        metavar="NAME:key=value;key=value",
        help="Inline scenario definition. Quote the full value in PowerShell.",
    )
    parser.add_argument(
        "--scenario-file",
        help=(
            "JSON file containing a list of scenarios with shape "
            '[{"name":"small","kwargs":{"grid_thw":"1,16,16","prompt_len":1}}]'
        ),
    )
    parser.add_argument(
        "--hip-runtime",
        help="Optional explicit path to the HIP runtime library (amdhip64)",
    )
    args = parser.parse_args()

    artifact_path = Path(args.artifact)
    module = _load_validation_module(args.validation_module)
    if not hasattr(module, "build_validation_inputs"):
        raise AttributeError(
            f"Validation module {args.validation_module!r} does not expose build_validation_inputs(**kwargs)"
        )

    metadata = json.loads((artifact_path / "metadata.json").read_text(encoding="utf-8"))
    common_kwargs = _parse_key_value_items(args.common_kwarg)
    scenarios = _load_scenarios(args.scenario, args.scenario_file)
    if not scenarios:
        raise ValueError("At least one scenario is required")
    built_scenarios = [
        {
            "name": scenario["name"],
            "kwargs": {**common_kwargs, **scenario["kwargs"]},
            "inputs": module.build_validation_inputs(**{**common_kwargs, **scenario["kwargs"]}),
        }
        for scenario in scenarios
    ]
    output_names = tuple(str(item["name"]) for item in metadata.get("outputs", []))
    hip_runtime = load_hip_runtime(args.hip_runtime)

    payload = {
        "artifact": str(artifact_path),
        "scenarios": [
            {
                "name": scenario["name"],
                "kwargs": scenario["kwargs"],
                "matching_bucket": _bucket_summary(metadata, scenario["inputs"]),
            }
            for scenario in built_scenarios
        ],
        "measurements": {
            "eager_max_create": _measure_eager_max_create(artifact_path, hip_runtime),
            "lazy_exact_bucket": {
                scenario["name"]: _measure_lazy_exact_bucket(artifact_path, hip_runtime, scenario["inputs"], output_names)
                for scenario in built_scenarios
            },
            "lazy_grow_sequence": _measure_lazy_grow_sequence(
                artifact_path,
                hip_runtime,
                built_scenarios,
                output_names,
            ),
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


def _load_validation_module(module_ref: str):
    candidate = Path(module_ref)
    if candidate.exists():
        spec = importlib.util.spec_from_file_location(candidate.stem, candidate)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load validation module from {candidate}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(module_ref)


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_key_value_items(items: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items:
        key, sep, value = str(item).partition("=")
        if not sep:
            raise ValueError(f"Expected KEY=VALUE, got {item!r}")
        key = key.strip()
        if not key:
            raise ValueError(f"Expected non-empty key in {item!r}")
        parsed[key] = _parse_scalar(value.strip())
    return parsed


def _parse_inline_scenario(text: str) -> dict[str, Any]:
    name, sep, raw_kwargs = text.partition(":")
    if not sep:
        raise ValueError(f"Expected scenario format NAME:key=value;key=value, got {text!r}")
    kwargs: dict[str, Any] = {}
    for item in raw_kwargs.split(";"):
        stripped = item.strip()
        if not stripped:
            continue
        kwargs.update(_parse_key_value_items([stripped]))
    return {"name": name.strip(), "kwargs": kwargs}


def _load_scenarios(inline_scenarios: list[str], scenario_file: str | None) -> list[dict[str, Any]]:
    scenarios = [_parse_inline_scenario(item) for item in inline_scenarios]
    if scenario_file is None:
        return scenarios
    payload = json.loads(Path(scenario_file).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Scenario file must contain a list")
    for item in payload:
        if not isinstance(item, Mapping):
            raise ValueError(f"Scenario entry must be an object, got {type(item).__name__}")
        scenarios.append(
            {
                "name": str(item["name"]),
                "kwargs": dict(item.get("kwargs", {})),
            }
        )
    return scenarios


def _sync_and_free_bytes(hip_runtime) -> int:
    hip_device_synchronize(hip_runtime)
    free, _total = hip_mem_get_info(hip_runtime)
    return free


def _measure_eager_max_create(artifact_path: Path, hip_runtime) -> dict[str, int]:
    module = runtime.load(artifact_path, load_constants=True)
    try:
        free_after_module = _sync_and_free_bytes(hip_runtime)
        session = module.create_session(temp_arena_policy="eager_max")
        try:
            free_after_session = _sync_and_free_bytes(hip_runtime)
            session.release_temp_arena()
            free_after_release = _sync_and_free_bytes(hip_runtime)
        finally:
            session.close()
    finally:
        module.close()
    return {
        "free_after_module": free_after_module,
        "free_after_session": free_after_session,
        "free_after_release": free_after_release,
        "observed_temp_bytes": free_after_release - free_after_session,
    }


def _measure_lazy_exact_bucket(
    artifact_path: Path,
    hip_runtime,
    inputs: Mapping[str, Any],
    output_names: tuple[str, ...],
) -> dict[str, int]:
    module = runtime.load(artifact_path, load_constants=True)
    try:
        free_after_module = _sync_and_free_bytes(hip_runtime)
        session = module.create_session(temp_arena_policy="lazy_exact_bucket")
        try:
            free_after_session = _sync_and_free_bytes(hip_runtime)
            session.run_numpy_device_outputs(inputs, device_outputs=output_names)
            free_after_run = _sync_and_free_bytes(hip_runtime)
            session._free_cuda_buffers()
            free_after_buffers = _sync_and_free_bytes(hip_runtime)
            session.release_temp_arena()
            free_after_release = _sync_and_free_bytes(hip_runtime)
        finally:
            session.close()
    finally:
        module.close()
    return {
        "free_after_module": free_after_module,
        "free_after_session": free_after_session,
        "free_after_run": free_after_run,
        "free_after_buffers": free_after_buffers,
        "free_after_release": free_after_release,
        "observed_temp_bytes": free_after_release - free_after_buffers,
        "observed_io_bytes": free_after_buffers - free_after_run,
    }


def _measure_lazy_grow_sequence(
    artifact_path: Path,
    hip_runtime,
    scenarios: list[dict[str, Any]],
    output_names: tuple[str, ...],
) -> dict[str, Any]:
    module = runtime.load(artifact_path, load_constants=True)
    try:
        free_after_module = _sync_and_free_bytes(hip_runtime)
        session = module.create_session(temp_arena_policy="lazy_grow")
        try:
            free_after_session = _sync_and_free_bytes(hip_runtime)
            points = []
            for scenario in scenarios:
                session.run_numpy_device_outputs(scenario["inputs"], device_outputs=output_names)
                _sync_and_free_bytes(hip_runtime)
                session._free_cuda_buffers()
                free_after_buffers = _sync_and_free_bytes(hip_runtime)
                points.append({"name": scenario["name"], "free_after_buffers": free_after_buffers})
            session.release_temp_arena()
            free_after_release = _sync_and_free_bytes(hip_runtime)
        finally:
            session.close()
    finally:
        module.close()
    return {
        "free_after_module": free_after_module,
        "free_after_session": free_after_session,
        "free_after_release": free_after_release,
        "points": [
            {
                **point,
                "observed_temp_bytes": free_after_release - point["free_after_buffers"],
            }
            for point in points
        ],
    }


def _bucket_summary(metadata: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any] | None:
    plan = matching_bucket_plan(metadata, inputs)
    if plan is None:
        return None
    return {
        "bucket_id": str(plan.get("bucket_id", "")),
        "arena_nbytes": int(plan.get("arena_nbytes", 0) or 0),
        "workspace_nbytes": int(plan.get("workspace_nbytes", 0) or 0),
    }


if __name__ == "__main__":
    raise SystemExit(main())
