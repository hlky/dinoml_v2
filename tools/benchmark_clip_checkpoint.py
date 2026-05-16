from __future__ import annotations

import argparse
import importlib
import json
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

import dinoml as dml
from dinoml import runtime

WORKFLOW_DIR = REPO_ROOT / "examples"
if str(WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_DIR))

import clip_checkpoint_workflow as clip_workflow


DEFAULT_BENCHMARK_DIR = REPO_ROOT / "tmp" / "clip_checkpoint_benchmark"
DEFAULT_JSON_OUT = DEFAULT_BENCHMARK_DIR / "clip_checkpoint_benchmark.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a cached Transformers CLIP checkpoint against a DinoML artifact."
    )
    parser.add_argument("--checkpoint-id", default=clip_workflow.DEFAULT_CHECKPOINT_ID)
    parser.add_argument("--target", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_JSON_OUT, help="JSON report path.")
    parser.add_argument("--warmup", type=_non_negative_int, default=1)
    parser.add_argument("--iters", type=_positive_int, default=3)
    parser.add_argument("--transformers-src", type=Path, default=clip_workflow.DEFAULT_TRANSFORMERS_SRC)
    parser.add_argument("--hf-home", type=Path, default=clip_workflow.DEFAULT_HF_HOME)
    return parser.parse_args(argv)


def run_benchmark(
    *,
    checkpoint_id: str = clip_workflow.DEFAULT_CHECKPOINT_ID,
    target: str = "cpu",
    artifact_dir: str | Path | None = None,
    out: str | Path | None = DEFAULT_JSON_OUT,
    warmup: int = 1,
    iters: int = 3,
    transformers_src: str | Path = clip_workflow.DEFAULT_TRANSFORMERS_SRC,
    hf_home: str | Path = clip_workflow.DEFAULT_HF_HOME,
) -> dict[str, object]:
    warmup = _non_negative_int(str(warmup))
    iters = _positive_int(str(iters))
    target = str(target)
    transformers_src = Path(transformers_src).resolve()
    hf_home = Path(hf_home).resolve()
    cache_dir = clip_workflow._ensure_local_dinoml_cache_dir()
    synchronize = _synchronize_for_target(target)

    if target == "cuda":
        if shutil.which("nvcc") is None:
            raise RuntimeError("nvcc is required for --target cuda")
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            raise RuntimeError("a CUDA device is required for --target cuda")

    clip_model = clip_workflow._load_cached_transformers_clip_checkpoint(
        checkpoint_id=checkpoint_id,
        transformers_src=transformers_src,
        hf_home=hf_home,
    )
    spec, inputs, text_config, vision_config = clip_workflow._trace_spec(clip_model=clip_model)

    target_spec = clip_workflow._target_spec(target)
    if artifact_dir is None:
        artifact_dir = DEFAULT_BENCHMARK_DIR / f"{_checkpoint_slug(checkpoint_id)}_{target}.dinoml"
    artifact_dir = Path(artifact_dir).resolve()

    compile_start = time.perf_counter()
    artifact = dml.compile(spec, target_spec, artifact_dir)
    compile_time_ms = (time.perf_counter() - compile_start) * 1000.0

    dinoml_result = _benchmark_dinoml_run_numpy(
        artifact_path=artifact.path,
        inputs=inputs,
        warmup=warmup,
        iters=iters,
        synchronize=synchronize,
    )
    transformers_result = _benchmark_transformers_forward(
        clip_model=clip_model,
        inputs=inputs,
        target=target,
        warmup=warmup,
        iters=iters,
        synchronize=synchronize,
    )

    limits = clip_workflow._limits_for(checkpoint_id=checkpoint_id, target=target)
    actual = dinoml_result["outputs"]
    expected = transformers_result["outputs"]
    parity = {
        name: clip_workflow._parity_entry(actual=actual[name], expected=expected[name], limit=limits[name])
        for name in clip_workflow.OUTPUT_NAMES
    }

    report: dict[str, object] = {
        "name": "clip_checkpoint_benchmark",
        "checkpoint_id": checkpoint_id,
        "target": target_spec.to_json(),
        "transformers_src": str(transformers_src),
        "hf_home": str(hf_home),
        "dinoml_cache_dir": str(cache_dir),
        "artifact": {
            "path": str(artifact.path),
            "module_exists": (artifact.path / "module.so").exists(),
            "manifest_exists": (artifact.path / "manifest.json").exists(),
        },
        "input_shapes": {name: list(value.shape) for name, value in inputs.items()},
        "output_shapes": {name: list(actual[name].shape) for name in clip_workflow.OUTPUT_NAMES},
        "limits": limits,
        "parity": parity,
        "allclose": {name: entry["allclose"] for name, entry in parity.items()},
        "max_abs_diff": {name: entry["max_abs_diff"] for name, entry in parity.items()},
        "timings_ms": {
            "compile": _single_timing(compile_time_ms),
            "runtime_load": dinoml_result["runtime_load"],
            "session_create": dinoml_result["session_create"],
            "dinoml_run_numpy": dinoml_result["latency"],
            "transformers_forward": transformers_result["latency"],
        },
        "text_config": {
            "max_position_embeddings": int(text_config.max_position_embeddings),
            "projection_dim": int(text_config.projection_dim),
            "eos_token_id": int(text_config.eos_token_id),
        },
        "vision_config": {
            "image_size": int(vision_config.image_size),
            "num_channels": int(vision_config.num_channels),
            "projection_dim": int(vision_config.projection_dim),
        },
        "benchmark": {
            "warmup": warmup,
            "iters": iters,
            "synchronized": target == "cuda",
            "inputs": "deterministic_synthetic_clip_checkpoint_workflow",
        },
    }

    if out is not None:
        out_path = Path(out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report["report_path"] = str(out_path)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    return report


def _benchmark_dinoml_run_numpy(
    *,
    artifact_path: Path,
    inputs: dict[str, np.ndarray],
    warmup: int,
    iters: int,
    synchronize: Callable[[], None],
) -> dict[str, object]:
    load_start = time.perf_counter()
    module = runtime.load(artifact_path)
    runtime_load = _single_timing((time.perf_counter() - load_start) * 1000.0)
    session_start = time.perf_counter()
    session = module.create_session()
    session_create = _single_timing((time.perf_counter() - session_start) * 1000.0)
    try:
        output_holder: dict[str, dict[str, np.ndarray]] = {}

        def run() -> None:
            output_holder["outputs"] = session.run_numpy(inputs)

        latency = benchmark_ms(run, warmup=warmup, iters=iters, synchronize=synchronize)
        run()
        return {
            "runtime_load": runtime_load,
            "session_create": session_create,
            "latency": latency,
            "outputs": output_holder["outputs"],
        }
    finally:
        session.close()
        module.close()


def _benchmark_transformers_forward(
    *,
    clip_model: object,
    inputs: dict[str, np.ndarray],
    target: str,
    warmup: int,
    iters: int,
    synchronize: Callable[[], None],
) -> dict[str, object]:
    torch = importlib.import_module("torch")
    device = "cuda" if target == "cuda" else "cpu"
    clip_model.to(device)
    clip_model.eval()
    torch_inputs = {
        name: torch.from_numpy(value).to(device=device)
        for name, value in inputs.items()
    }
    output_holder: dict[str, object] = {}

    def run() -> None:
        with torch.inference_mode():
            output_holder["outputs"] = clip_model(
                input_ids=torch_inputs["input_ids"],
                attention_mask=torch_inputs["attention_mask"],
                pixel_values=torch_inputs["pixel_values"],
            )

    latency = benchmark_ms(run, warmup=warmup, iters=iters, synchronize=synchronize)
    run()
    outputs = output_holder["outputs"]
    return {
        "latency": latency,
        "outputs": {
            "logits_per_image": outputs.logits_per_image.detach().cpu().numpy().astype(np.float32),
            "logits_per_text": outputs.logits_per_text.detach().cpu().numpy().astype(np.float32),
            "text_embeds": outputs.text_embeds.detach().cpu().numpy().astype(np.float32),
            "image_embeds": outputs.image_embeds.detach().cpu().numpy().astype(np.float32),
        },
    }


def benchmark_ms(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    synchronize: Callable[[], None] | None = None,
) -> dict[str, float | int]:
    if synchronize is None:
        synchronize = lambda: None
    for _ in range(warmup):
        fn()
    synchronize()
    samples = []
    for _ in range(iters):
        synchronize()
        start = time.perf_counter()
        fn()
        synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return summarize_ms(samples, warmup=warmup)


def summarize_ms(samples: list[float], *, warmup: int = 0) -> dict[str, float | int]:
    if not samples:
        raise ValueError("at least one timing sample is required")
    return {
        "count": len(samples),
        "warmup": warmup,
        "mean": float(statistics.fmean(samples)),
        "median": float(statistics.median(samples)),
        "min": float(min(samples)),
        "max": float(max(samples)),
        "stddev": float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0,
    }


def _single_timing(elapsed_ms: float) -> dict[str, float | int]:
    return {
        "count": 1,
        "warmup": 0,
        "mean": float(elapsed_ms),
        "median": float(elapsed_ms),
        "min": float(elapsed_ms),
        "max": float(elapsed_ms),
        "stddev": 0.0,
    }


def _synchronize_for_target(target: str) -> Callable[[], None]:
    if target != "cuda":
        return lambda: None
    torch = importlib.import_module("torch")

    def synchronize() -> None:
        torch.cuda.synchronize()

    return synchronize


def _checkpoint_slug(checkpoint_id: str) -> str:
    return checkpoint_id.replace("/", "__").replace("-", "_")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_benchmark(
        checkpoint_id=args.checkpoint_id,
        target=args.target,
        artifact_dir=args.artifact_dir,
        out=args.out,
        warmup=args.warmup,
        iters=args.iters,
        transformers_src=args.transformers_src,
        hf_home=args.hf_home,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
