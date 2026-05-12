import json
import os
import subprocess
import sys
from pathlib import Path

from dinoml import runtime


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = "examples/fused_elementwise.py"
IMAGE_POOLING_EXAMPLE = "examples/image_pooling.py"
CANDIDATE_SELECTION_EXAMPLE = "examples/candidate_selection.py"


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = src_path
    result = subprocess.run(
        [sys.executable, "-m", "dinoml.cli", *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_cpu_cli_compile_inspect_validate_quick_start(tmp_path):
    artifact = tmp_path / "fused_elementwise_cpu.dinoml"

    compile_result = _run_cli("compile", EXAMPLE, "--target", "cpu", "--out", str(artifact))
    assert f"Wrote {artifact}" in compile_result.stdout
    assert (artifact / "module.so").exists()

    inspect_result = _run_cli("inspect", str(artifact))
    summary = json.loads(inspect_result.stdout)
    assert summary["name"] == "fused_elementwise"
    assert summary["target"]["name"] == "cpu"
    assert summary["inputs"][0]["name"] == "x"
    assert summary["outputs"][0]["name"] == "y"
    assert summary["nodes"] == 1
    assert summary["constants"] == 3

    validate_result = _run_cli("validate", str(artifact), "--against", EXAMPLE)
    assert "y: max_abs_diff=" in validate_result.stdout
    assert "validation ok" in validate_result.stdout


def test_cpu_cli_deferred_constant_workflow(tmp_path):
    artifact = tmp_path / "deferred_constants_cpu.dinoml"

    compile_result = _run_cli(
        "compile",
        EXAMPLE,
        "--target",
        "cpu",
        "--constant-load-policy",
        "deferred",
        "--out",
        str(artifact),
    )
    assert f"Wrote {artifact}" in compile_result.stdout

    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["constant_load_policy"] == "deferred"

    module = runtime.load(artifact)
    try:
        load_state = module.constant_load_state()
        assert load_state["scale"] is False
        assert load_state["bias"] is False
        assert all(loaded is False for loaded in load_state.values())
    finally:
        module.close()

    validate_result = _run_cli("validate", str(artifact), "--against", EXAMPLE)
    assert "y: max_abs_diff=" in validate_result.stdout
    assert "validation ok" in validate_result.stdout


def test_cpu_cli_image_pooling_example_compile_inspect_validate(tmp_path):
    artifact = tmp_path / "image_pooling_cpu.dinoml"

    compile_result = _run_cli("compile", IMAGE_POOLING_EXAMPLE, "--target", "cpu", "--out", str(artifact))
    assert f"Wrote {artifact}" in compile_result.stdout
    assert (artifact / "module.so").exists()

    inspect_result = _run_cli("inspect", str(artifact))
    summary = json.loads(inspect_result.stdout)
    assert summary["name"] == "image_pooling"
    assert summary["target"]["name"] == "cpu"
    assert summary["inputs"][0]["name"] == "x"
    assert summary["outputs"][0]["name"] == "features"
    assert [summary["outputs"][0]["shape"], summary["outputs"][0]["dtype"]] == [[1, 1, 4, 4], "float32"]
    assert summary["nodes"] == 3
    assert summary["constants"] == 0

    validate_result = _run_cli("validate", str(artifact), "--against", IMAGE_POOLING_EXAMPLE)
    assert "features: max_abs_diff=" in validate_result.stdout
    assert "validation ok" in validate_result.stdout


def test_cpu_cli_candidate_selection_example_compile_inspect_validate(tmp_path):
    artifact = tmp_path / "candidate_selection_cpu.dinoml"

    compile_result = _run_cli(
        "compile", CANDIDATE_SELECTION_EXAMPLE, "--target", "cpu", "--out", str(artifact)
    )
    assert f"Wrote {artifact}" in compile_result.stdout
    assert (artifact / "module.so").exists()

    inspect_result = _run_cli("inspect", str(artifact))
    summary = json.loads(inspect_result.stdout)
    assert summary["name"] == "candidate_selection"
    assert summary["target"]["name"] == "cpu"
    assert [input_spec["name"] for input_spec in summary["inputs"]] == [
        "scores",
        "features",
    ]
    assert [output["name"] for output in summary["outputs"]] == [
        "top_scores",
        "selected_features",
    ]
    assert [output["shape"] for output in summary["outputs"]] == [[2, 2], [2, 2, 3]]
    assert [output["dtype"] for output in summary["outputs"]] == ["float32", "float32"]
    assert summary["nodes"] == 3
    assert summary["constants"] == 0

    validate_result = _run_cli(
        "validate", str(artifact), "--against", CANDIDATE_SELECTION_EXAMPLE
    )
    assert "top_scores: max_abs_diff=" in validate_result.stdout
    assert "selected_features: max_abs_diff=" in validate_result.stdout
    assert "validation ok" in validate_result.stdout
