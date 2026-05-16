import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "benchmark_clip_checkpoint.py"
WORKFLOW = REPO_ROOT / "examples" / "clip_checkpoint_workflow.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("benchmark_clip_checkpoint_under_test", TOOL)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fake_timing() -> dict[str, float | int]:
    return {
        "count": 1,
        "warmup": 0,
        "mean": 0.25,
        "median": 0.25,
        "min": 0.25,
        "max": 0.25,
        "stddev": 0.0,
    }


def test_clip_checkpoint_benchmark_cli_parses_bounded_options(tmp_path):
    tool = _load_tool()
    args = tool.parse_args(
        [
            "--checkpoint-id",
            "local/clip",
            "--target",
            "cuda",
            "--artifact-dir",
            str(tmp_path / "artifact.dinoml"),
            "--out",
            str(tmp_path / "report.json"),
            "--warmup",
            "2",
            "--iters",
            "5",
            "--transformers-src",
            "/workspace/transformers/src",
            "--hf-home",
            "/workspace/.cache/huggingface",
        ]
    )

    assert args.checkpoint_id == "local/clip"
    assert args.target == "cuda"
    assert args.artifact_dir == tmp_path / "artifact.dinoml"
    assert args.out == tmp_path / "report.json"
    assert args.warmup == 2
    assert args.iters == 5


def test_clip_checkpoint_benchmark_summary_schema_with_stubbed_runtime(monkeypatch, tmp_path):
    tool = _load_tool()
    clip_workflow = tool.clip_workflow
    dml = tool.dml

    inputs = {
        "input_ids": np.asarray([[0, 1, 3, 31]], dtype=np.int64),
        "attention_mask": np.ones((1, 4), dtype=np.bool_),
        "pixel_values": np.zeros((1, 3, 4, 4), dtype=np.float32),
    }
    outputs = {
        "logits_per_image": np.asarray([[1.0]], dtype=np.float32),
        "logits_per_text": np.asarray([[1.0]], dtype=np.float32),
        "text_embeds": np.zeros((1, 8), dtype=np.float32),
        "image_embeds": np.zeros((1, 8), dtype=np.float32),
    }
    artifact_dir = tmp_path / "clip.dinoml"

    monkeypatch.setattr(clip_workflow, "_ensure_local_dinoml_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(clip_workflow, "_load_cached_transformers_clip_checkpoint", lambda **_: object())
    monkeypatch.setattr(
        clip_workflow,
        "_trace_spec",
        lambda **_: (
            object(),
            inputs,
            SimpleNamespace(max_position_embeddings=77, projection_dim=8, eos_token_id=2),
            SimpleNamespace(image_size=4, num_channels=3, projection_dim=8),
        ),
    )

    def fake_compile(spec, target, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "module.so").write_bytes(b"")
        (path / "manifest.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(path=path)

    monkeypatch.setattr(dml, "compile", fake_compile)
    monkeypatch.setattr(tool, "_synchronize_for_target", lambda target: lambda: None)
    monkeypatch.setattr(
        tool,
        "_benchmark_dinoml_run_numpy",
        lambda **_: {
            "runtime_load": _fake_timing(),
            "session_create": _fake_timing(),
            "latency": _fake_timing(),
            "outputs": outputs,
        },
    )
    monkeypatch.setattr(
        tool,
        "_benchmark_transformers_forward",
        lambda **_: {
            "latency": _fake_timing(),
            "outputs": outputs,
        },
    )

    report = tool.run_benchmark(
        checkpoint_id="openai/clip-vit-base-patch32",
        target="cpu",
        artifact_dir=artifact_dir,
        out=tmp_path / "report.json",
        warmup=0,
        iters=1,
    )

    assert report["name"] == "clip_checkpoint_benchmark"
    assert report["checkpoint_id"] == "openai/clip-vit-base-patch32"
    assert report["target"]["name"] == "cpu"
    assert report["artifact"]["module_exists"] is True
    assert report["input_shapes"] == {
        "attention_mask": [1, 4],
        "input_ids": [1, 4],
        "pixel_values": [1, 3, 4, 4],
    }
    assert report["output_shapes"]["logits_per_image"] == [1, 1]
    assert set(report["timings_ms"]) == {
        "compile",
        "runtime_load",
        "session_create",
        "dinoml_run_numpy",
        "transformers_forward",
    }
    assert all(report["allclose"].values())
    written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert written["report_path"] == str((tmp_path / "report.json").resolve())


def test_clip_checkpoint_benchmark_timing_summary_helpers():
    tool = _load_tool()
    calls = {"count": 0}

    def fn():
        calls["count"] += 1

    summary = tool.benchmark_ms(fn, warmup=2, iters=3)
    assert calls["count"] == 5
    assert summary["count"] == 3
    assert summary["warmup"] == 2
    assert summary["min"] <= summary["median"] <= summary["max"]


@pytest.mark.filterwarnings("ignore:overflow encountered in exp:RuntimeWarning")
def test_clip_checkpoint_benchmark_cli_cpu_smoke_cached_base(tmp_path):
    workflow = _load_workflow()
    available, reason = workflow._checkpoint_is_available(
        checkpoint_id="openai/clip-vit-base-patch32",
        transformers_src=Path("/workspace/transformers/src"),
        hf_home=Path("/workspace/.cache/huggingface"),
    )
    if not available:
        pytest.skip(f"cached base checkpoint benchmark unavailable: {reason}")

    artifact_dir = tmp_path / "clip_checkpoint_benchmark_cpu.dinoml"
    report_path = tmp_path / "benchmark.json"
    env = os.environ.copy()
    env["DINOML_CACHE_DIR"] = str(tmp_path / "cache")
    env["HF_HOME"] = "/workspace/.cache/huggingface"
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--artifact-dir",
            str(artifact_dir),
            "--out",
            str(report_path),
            "--warmup",
            "0",
            "--iters",
            "1",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    summary = json.loads(result.stdout)
    assert summary["name"] == "clip_checkpoint_benchmark"
    assert summary["checkpoint_id"] == "openai/clip-vit-base-patch32"
    assert summary["target"]["name"] == "cpu"
    assert summary["artifact"]["path"] == str(artifact_dir.resolve())
    assert summary["report_path"] == str(report_path.resolve())
    assert summary["benchmark"] == {
        "inputs": "deterministic_synthetic_clip_checkpoint_workflow",
        "iters": 1,
        "synchronized": False,
        "warmup": 0,
    }
    assert all(summary["allclose"].values())
    assert all(metric <= summary["limits"][name] for name, metric in summary["max_abs_diff"].items())
    for key in ("compile", "runtime_load", "session_create", "dinoml_run_numpy", "transformers_forward"):
        assert summary["timings_ms"][key]["count"] >= 1


def _load_workflow():
    spec = importlib.util.spec_from_file_location("clip_checkpoint_workflow_under_test", WORKFLOW)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
