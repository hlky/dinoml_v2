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


def test_clip_checkpoint_benchmark_cuda_report_includes_hot_path_timings(monkeypatch, tmp_path):
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

    def fake_timing(mean: float) -> dict[str, float | int]:
        return {
            "count": 2,
            "warmup": 1,
            "mean": mean,
            "median": mean,
            "min": mean,
            "max": mean,
            "stddev": 0.0,
        }

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
    monkeypatch.setattr(clip_workflow, "_target_spec", lambda target: SimpleNamespace(to_json=lambda: {"name": target}))
    monkeypatch.setattr(clip_workflow, "_limits_for", lambda **_: {name: 1e-5 for name in tool.clip_workflow.OUTPUT_NAMES})
    monkeypatch.setattr(
        clip_workflow,
        "_parity_entry",
        lambda *, actual, expected, limit: {"allclose": True, "max_abs_diff": 0.0, "limit": limit},
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
            "latency": fake_timing(4.0),
            "outputs": outputs,
        },
    )
    monkeypatch.setattr(
        tool,
        "_benchmark_dinoml_cuda_hot_paths",
        lambda **_: {
            "run_device_pointers": fake_timing(1.5),
            "output_shapes": {name: list(value.shape) for name, value in outputs.items()},
            "run_torch_error": "ValueError: Unsupported torch dtype: torch.int64",
        },
    )
    monkeypatch.setattr(
        tool,
        "_benchmark_transformers_forward",
        lambda **_: {
            "latency": fake_timing(3.0),
            "outputs": outputs,
        },
    )

    report = tool.run_benchmark(
        checkpoint_id="openai/clip-vit-base-patch32",
        target="cuda",
        artifact_dir=tmp_path / "clip_cuda.dinoml",
        out=None,
        warmup=1,
        iters=2,
    )

    assert report["target"]["name"] == "cuda"
    assert report["timings_ms"]["dinoml_run_numpy"]["mean"] == pytest.approx(4.0)
    assert report["timings_ms"]["dinoml_run_device_pointers"]["mean"] == pytest.approx(1.5)
    assert report["cuda_run_numpy_overhead_ms"] == {
        "run_device_pointers": {"mean": 2.5, "median": 2.5},
    }
    assert report["cuda_run_torch"] == {
        "available": False,
        "error": "ValueError: Unsupported torch dtype: torch.int64",
    }


def test_clip_checkpoint_benchmark_cuda_hot_path_helper_uses_runtime_entrypoints(monkeypatch, tmp_path):
    tool = _load_tool()

    class FakeTensor:
        _next_ptr = 0x1000

        def __init__(self, shape, *, device="cuda"):
            self.shape = tuple(shape)
            self.device = device
            self._ptr = FakeTensor._next_ptr
            FakeTensor._next_ptr += 0x100

        def to(self, *, device):
            return FakeTensor(self.shape, device=device)

        def data_ptr(self):
            return self._ptr

    class FakeTorchModule:
        float32 = "float32"
        float16 = "float16"
        bfloat16 = "bfloat16"
        int64 = "int64"
        int32 = "int32"
        bool = "bool"

        def from_numpy(self, value):
            return FakeTensor(value.shape, device="cpu")

        def empty(self, shape, *, dtype, device):
            return FakeTensor(shape, device=device)

    calls = []

    class FakeSession:
        def run_torch(self, inputs):
            calls.append(("run_torch", {name: tuple(value.shape) for name, value in inputs.items()}))
            raise ValueError("Unsupported torch dtype: torch.int64")

        def run_device_pointers(self, inputs, outputs, input_shapes, output_shapes):
            calls.append(("run_device_pointers", dict(inputs), dict(outputs), dict(input_shapes), dict(output_shapes)))

        def close(self):
            calls.append(("session_close",))

    class FakeModule:
        metadata = {
            "outputs": [
                {"name": "logits_per_image", "dtype": "float32"},
                {"name": "logits_per_text", "dtype": "float32"},
                {"name": "text_embeds", "dtype": "float32"},
                {"name": "image_embeds", "dtype": "float32"},
            ]
        }

        def create_session(self):
            calls.append(("create_session",))
            return FakeSession()

        def close(self):
            calls.append(("module_close",))

    latencies = iter(
        [
            {
                "count": 3,
                "warmup": 1,
                "mean": 1.0,
                "median": 1.1,
                "min": 0.9,
                "max": 1.2,
                "stddev": 0.1,
            },
        ]
    )

    def fake_benchmark_ms(fn, *, warmup, iters, synchronize):
        fn()
        return next(latencies)

    monkeypatch.setattr(tool.runtime, "load", lambda path: FakeModule())
    monkeypatch.setattr(tool.importlib, "import_module", lambda name: FakeTorchModule() if name == "torch" else importlib.import_module(name))
    monkeypatch.setattr(tool, "benchmark_ms", fake_benchmark_ms)

    report = tool._benchmark_dinoml_cuda_hot_paths(
        artifact_path=tmp_path / "artifact.dinoml",
        inputs={
            "input_ids": np.zeros((1, 4), dtype=np.int64),
            "attention_mask": np.ones((1, 4), dtype=np.bool_),
            "pixel_values": np.zeros((1, 3, 4, 4), dtype=np.float32),
        },
        output_shapes={
            "logits_per_image": (1, 1),
            "logits_per_text": (1, 1),
            "text_embeds": (1, 8),
            "image_embeds": (1, 8),
        },
        warmup=1,
        iters=3,
        synchronize=lambda: None,
    )

    assert report["run_device_pointers"]["mean"] == pytest.approx(1.0)
    assert report["output_shapes"] == {
        "logits_per_image": [1, 1],
        "logits_per_text": [1, 1],
        "text_embeds": [1, 8],
        "image_embeds": [1, 8],
    }
    assert report["run_torch_error"] == "ValueError: Unsupported torch dtype: torch.int64"
    assert calls[0] == ("create_session",)
    assert calls[1][0] == "run_device_pointers"
    assert calls[1][3] == {
        "input_ids": (1, 4),
        "attention_mask": (1, 4),
        "pixel_values": (1, 3, 4, 4),
    }
    assert calls[1][4] == {
        "logits_per_image": (1, 1),
        "logits_per_text": (1, 1),
        "text_embeds": (1, 8),
        "image_embeds": (1, 8),
    }
    assert calls[2][0] == "run_torch"
    assert calls[-2:] == [("session_close",), ("module_close",)]


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
