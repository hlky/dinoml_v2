import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dinoml.reference import reference_numpy


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "clip_text_workflow.py"


def _load_example() -> dict[str, object]:
    return runpy.run_path(str(EXAMPLE))


@pytest.mark.parametrize(
    ("eos_token_id", "expected_eq_count", "expects_integer_eq_kernel"),
    [
        (2, 0, False),
        (7, 1, True),
    ],
)
def test_clip_text_workflow_example_proves_wrapper_and_artifact_state(
    eos_token_id, expected_eq_count, expects_integer_eq_kernel
):
    example = _load_example()
    spec = example["build_spec"](eos_token_id=eos_token_id)
    inputs = example["build_validation_inputs"](eos_token_id=eos_token_id)

    actual = reference_numpy(spec, inputs)["text_features"]
    expected = example["reference_outputs"](eos_token_id=eos_token_id)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    summary = example["inspect_workflow"](eos_token_id=eos_token_id)
    assert summary["name"] == f"clip_text_workflow_eos_{eos_token_id}"
    assert summary["output_name"] == "text_features"
    assert summary["output_shape"] == [example["BATCH"], example["PROJECTION"]]
    assert summary["node_op_counts"]["embedding"] == 2
    assert summary["node_op_counts"]["layer_norm"] == 3
    assert summary["node_op_counts"]["gemm_rcr_bias"] == 5
    assert summary["node_op_counts"]["gemm_rcr_bias_quick_gelu"] == 1
    assert summary["node_op_counts"]["gemm_rcr"] == 1
    assert summary["node_op_counts"]["bmm_rcr"] == 1
    assert summary["node_op_counts"]["bmm_rrr"] == 1
    assert summary["node_op_counts"].get("eq", 0) == expected_eq_count
    assert summary["uses_batch_gather_pooling"] is True
    assert summary["provider_kernel_ops"] == [
        "bmm_rcr",
        "bmm_rrr",
        "gemm_rcr",
        "gemm_rcr_bias",
        "gemm_rcr_bias_quick_gelu",
    ]
    assert summary["provider_kernel_libraries"] == ["cutlass_bmm", "cutlass_gemm"]
    assert "model" in summary["required_kernel_libraries"]
    assert summary["generated_cuda_kernel_count"] >= 7
    assert summary["has_integer_eos_eq_kernel"] is expects_integer_eq_kernel


def test_clip_text_workflow_example_script_smoke():
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = src_path
    result = subprocess.run(
        [sys.executable, str(EXAMPLE), "--eos-token-id", "7"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    summary = json.loads(result.stdout)
    assert summary["eos_token_id"] == 7
    assert summary["output_name"] == "text_features"
    assert summary["uses_batch_gather_pooling"] is True
    assert summary["has_integer_eos_eq_kernel"] is True
    assert len(summary["text_features"]) == 2
