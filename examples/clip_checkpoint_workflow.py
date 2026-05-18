from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

import numpy as np

import dinoml as dml
from dinoml import runtime
from dinoml.models.clip import (
    legacy_clip_configs_from_transformers_clip_config,
    legacy_clip_model_from_transformers_clip_model,
)


DEFAULT_CHECKPOINT_ID = "openai/clip-vit-base-patch32"
DEFAULT_HF_HOME = Path("/workspace/.cache/huggingface")
DEFAULT_TRANSFORMERS_SRC = Path("/workspace/transformers/src")
DEFAULT_CUDA_ARCH = "sm_86"
RTOL = 1.0e-5
OUTPUT_NAMES = ("logits_per_image", "logits_per_text", "text_embeds", "image_embeds")


def _artifact_dir_for(checkpoint_id: str, target: str) -> Path:
    checkpoint_slug = checkpoint_id.replace("/", "__").replace("-", "_")
    return (REPO_ROOT / "tmp" / f"clip_checkpoint_workflow_{checkpoint_slug}_{target}.dinoml").resolve()


def _target_spec(target: str) -> dml.Target:
    if target == "cpu":
        return dml.Target("cpu")
    if target == "cuda":
        return dml.Target("cuda", arch=DEFAULT_CUDA_ARCH, no_tf32=True)
    raise ValueError(f"unsupported target {target!r}")


def _limits_for(*, checkpoint_id: str, target: str) -> dict[str, float]:
    if target == "cpu":
        return {name: 3.0e-5 for name in OUTPUT_NAMES}
    limits = {
        "logits_per_image": 1.0e-5,
        "logits_per_text": 1.0e-5,
        "text_embeds": 1.0e-5,
        "image_embeds": 1.0e-5,
    }
    if checkpoint_id == "openai/clip-vit-large-patch14":
        limits["logits_per_image"] = 2.0e-5
        limits["logits_per_text"] = 2.0e-5
    return limits


def _import_local_transformers(transformers_src: Path):
    transformers_src = Path(transformers_src).resolve()
    if not transformers_src.exists():
        raise FileNotFoundError(f"Transformers source tree not found: {transformers_src}")
    loaded = sys.modules.get("transformers")
    if loaded is not None:
        loaded_path = Path(getattr(loaded, "__file__", "")).resolve()
        if not loaded_path.is_relative_to(transformers_src):
            for name in list(sys.modules):
                if name == "transformers" or name.startswith("transformers."):
                    del sys.modules[name]
    if str(transformers_src) not in sys.path:
        sys.path.insert(0, str(transformers_src))
    transformers = importlib.import_module("transformers")
    resolved = Path(transformers.__file__).resolve()
    if not resolved.is_relative_to(transformers_src):
        raise RuntimeError(f"expected local Transformers import from {transformers_src}, got {resolved}")
    return transformers


def _load_cached_transformers_clip_checkpoint(
    *,
    checkpoint_id: str,
    transformers_src: Path,
    hf_home: Path,
):
    os.environ["HF_HOME"] = str(Path(hf_home).resolve())
    transformers = _import_local_transformers(transformers_src)
    clip_model = transformers.CLIPModel.from_pretrained(checkpoint_id, local_files_only=True)
    clip_model.eval()
    return clip_model


def _checkpoint_is_available(
    *,
    checkpoint_id: str,
    transformers_src: Path,
    hf_home: Path,
) -> tuple[bool, str | None]:
    try:
        _load_cached_transformers_clip_checkpoint(
            checkpoint_id=checkpoint_id,
            transformers_src=transformers_src,
            hf_home=hf_home,
        )
    except Exception as exc:
        return False, str(exc)
    return True, None


def _build_runtime_inputs(*, text_config, vision_config) -> tuple[int, dict[str, np.ndarray]]:
    seq_len = min(4, int(text_config.max_position_embeddings))
    eos_token_id = int(text_config.eos_token_id)
    vocab_size = int(text_config.vocab_size)

    token_ids: list[int] = []
    candidate = 0
    while len(token_ids) < max(seq_len - 1, 0):
        if candidate != eos_token_id:
            token_ids.append(candidate)
        candidate += 1
    token_ids.append(vocab_size - 1 if eos_token_id == 2 else eos_token_id)

    image_size = int(vision_config.image_size)
    num_channels = int(vision_config.num_channels)
    pixel_values = np.linspace(
        -1.0,
        1.0,
        num=num_channels * image_size * image_size,
        dtype=np.float32,
    ).reshape(1, num_channels, image_size, image_size)

    return seq_len, {
        "input_ids": np.asarray([token_ids], dtype=np.int64),
        "attention_mask": np.ones((1, seq_len), dtype=np.bool_),
        "pixel_values": pixel_values,
    }


def _trace_spec(*, clip_model) -> tuple[dml.ir.ModelSpec, dict[str, np.ndarray], object, object]:
    text_config, vision_config = legacy_clip_configs_from_transformers_clip_config(clip_model.config)
    adapted_model = legacy_clip_model_from_transformers_clip_model(clip_model)
    seq_len, inputs = _build_runtime_inputs(text_config=text_config, vision_config=vision_config)
    spec = dml.trace(
        adapted_model,
        inputs={
            "input_ids": dml.TensorSpec([1, seq_len], "int64"),
            "pixel_values": dml.TensorSpec(
                [1, int(vision_config.num_channels), int(vision_config.image_size), int(vision_config.image_size)],
                "float32",
            ),
            "attention_mask": dml.TensorSpec([1, seq_len], "bool"),
        },
        name="clip_checkpoint_workflow",
    )
    return spec, inputs, text_config, vision_config


def _expected_outputs(*, clip_model, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    torch = importlib.import_module("torch")
    with torch.inference_mode():
        expected = clip_model(
            input_ids=torch.from_numpy(inputs["input_ids"]),
            attention_mask=torch.from_numpy(inputs["attention_mask"]),
            pixel_values=torch.from_numpy(inputs["pixel_values"]),
        )
    return {
        "logits_per_image": expected.logits_per_image.detach().cpu().numpy().astype(np.float32),
        "logits_per_text": expected.logits_per_text.detach().cpu().numpy().astype(np.float32),
        "text_embeds": expected.text_embeds.detach().cpu().numpy().astype(np.float32),
        "image_embeds": expected.image_embeds.detach().cpu().numpy().astype(np.float32),
    }


def _run_artifact(*, artifact_path: Path, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    module = runtime.load(artifact_path)
    session = module.create_session()
    try:
        return session.run_numpy(inputs)
    finally:
        session.close()
        module.close()


def _parity_entry(*, actual: np.ndarray, expected: np.ndarray, limit: float) -> dict[str, object]:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    diff = np.abs(actual - expected)
    max_abs_diff = float(diff.max()) if diff.size else 0.0
    return {
        "allclose": bool(np.allclose(actual, expected, atol=limit, rtol=RTOL)),
        "limit": limit,
        "max_abs_diff": max_abs_diff,
    }


def run_workflow(
    *,
    checkpoint_id: str = DEFAULT_CHECKPOINT_ID,
    target: str = "cpu",
    artifact_dir: str | Path | None = None,
    transformers_src: str | Path = DEFAULT_TRANSFORMERS_SRC,
    hf_home: str | Path = DEFAULT_HF_HOME,
) -> dict[str, object]:
    target = str(target)
    transformers_src = Path(transformers_src).resolve()
    hf_home = Path(hf_home).resolve()
    cache_dir = Path(os.environ.get("DINOML_CACHE_DIR", Path.home() / ".cache" / "dinoml_v2")).resolve()
    if target == "cuda":
        if shutil.which("nvcc") is None:
            raise RuntimeError("nvcc is required for --target cuda")
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            raise RuntimeError("a CUDA device is required for --target cuda")

    clip_model = _load_cached_transformers_clip_checkpoint(
        checkpoint_id=checkpoint_id,
        transformers_src=transformers_src,
        hf_home=hf_home,
    )
    spec, inputs, text_config, vision_config = _trace_spec(clip_model=clip_model)
    expected = _expected_outputs(clip_model=clip_model, inputs=inputs)

    target_spec = _target_spec(target)
    artifact_dir = _artifact_dir_for(checkpoint_id, target) if artifact_dir is None else Path(artifact_dir).resolve()
    artifact = dml.compile(spec, target_spec, artifact_dir)
    actual = _run_artifact(artifact_path=artifact.path, inputs=inputs)

    limits = _limits_for(checkpoint_id=checkpoint_id, target=target)
    parity = {
        name: _parity_entry(actual=actual[name], expected=expected[name], limit=limits[name])
        for name in OUTPUT_NAMES
    }
    output_shapes = {name: list(actual[name].shape) for name in OUTPUT_NAMES}
    return {
        "name": "clip_checkpoint_workflow",
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
        "output_shapes": output_shapes,
        "limits": limits,
        "parity": parity,
        "allclose": {name: entry["allclose"] for name, entry in parity.items()},
        "max_abs_diff": {name: entry["max_abs_diff"] for name, entry in parity.items()},
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile and run a cached Transformers CLIP checkpoint through DinoML."
    )
    parser.add_argument("--checkpoint-id", default=DEFAULT_CHECKPOINT_ID)
    parser.add_argument("--target", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--transformers-src", type=Path, default=DEFAULT_TRANSFORMERS_SRC)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    args = parser.parse_args()
    summary = run_workflow(
        checkpoint_id=args.checkpoint_id,
        target=args.target,
        artifact_dir=args.artifact_dir,
        transformers_src=args.transformers_src,
        hf_home=args.hf_home,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
