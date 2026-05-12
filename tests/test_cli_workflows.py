import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = "examples/fused_elementwise.py"


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
