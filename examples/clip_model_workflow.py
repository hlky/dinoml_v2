from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

import numpy as np

import dinoml as dml
from dinoml import runtime
from dinoml.backends.cpu import execute_cpu
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.lowering.ops import collect_generated_sources
from dinoml.models.clip import LegacyCLIPModel, LegacyCLIPTextConfig, LegacyCLIPVisionConfig
from dinoml.passes import PassManager, validate_ir


TEXT_BATCH = 2
IMAGE_BATCH = 3
SEQ_LEN = 4
VOCAB_SIZE = 16
TEXT_HIDDEN = 6
VISION_HIDDEN = 6
NUM_HEADS = 2
TEXT_INTERMEDIATE = 8
VISION_INTERMEDIATE = 10
PROJECTION = 5
EPS = 1.0e-5
MAX_POSITION_EMBEDDINGS = 6
NUM_CHANNELS = 3
IMAGE_SIZE = 4
PATCH_SIZE = 2
CUDA_TARGET = {"name": "cuda", "arch": "sm_86"}
LOCAL_TRANSFORMERS_SRC = Path("/workspace/transformers/src")
ATOL = 1.0e-5
RTOL = 1.0e-5
MODEL_OUTPUT_NAMES = ("logits_per_image", "logits_per_text", "text_embeds", "image_embeds")


def build_text_config() -> LegacyCLIPTextConfig:
    return LegacyCLIPTextConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        hidden_size=TEXT_HIDDEN,
        intermediate_size=TEXT_INTERMEDIATE,
        num_attention_heads=NUM_HEADS,
        num_hidden_layers=1,
        projection_dim=PROJECTION,
        layer_norm_eps=EPS,
        eos_token_id=2,
    )


def build_vision_config() -> LegacyCLIPVisionConfig:
    return LegacyCLIPVisionConfig(
        hidden_size=VISION_HIDDEN,
        intermediate_size=VISION_INTERMEDIATE,
        num_attention_heads=NUM_HEADS,
        num_hidden_layers=1,
        projection_dim=PROJECTION,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_channels=NUM_CHANNELS,
        layer_norm_eps=EPS,
    )


def build_weights() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(2089)

    def _normal(shape, scale):
        return (rng.standard_normal(shape).astype(np.float32) / scale).astype(np.float32)

    text_config = build_text_config()
    vision_config = build_vision_config()
    return {
        "text_model.embeddings.token_embedding.weight": _normal((text_config.vocab_size, text_config.hidden_size), 3.5),
        "text_model.embeddings.position_embedding.weight": _normal(
            (text_config.max_position_embeddings, text_config.hidden_size), 4.0
        ),
        "text_model.encoder.layers.0.self_attn.q_proj.weight": _normal(
            (text_config.hidden_size, text_config.hidden_size), 5.0
        ),
        "text_model.encoder.layers.0.self_attn.q_proj.bias": _normal((text_config.hidden_size,), 7.0),
        "text_model.encoder.layers.0.self_attn.k_proj.weight": _normal(
            (text_config.hidden_size, text_config.hidden_size), 5.0
        ),
        "text_model.encoder.layers.0.self_attn.k_proj.bias": _normal((text_config.hidden_size,), 7.0),
        "text_model.encoder.layers.0.self_attn.v_proj.weight": _normal(
            (text_config.hidden_size, text_config.hidden_size), 5.0
        ),
        "text_model.encoder.layers.0.self_attn.v_proj.bias": _normal((text_config.hidden_size,), 7.0),
        "text_model.encoder.layers.0.self_attn.out_proj.weight": _normal(
            (text_config.hidden_size, text_config.hidden_size), 5.0
        ),
        "text_model.encoder.layers.0.self_attn.out_proj.bias": _normal((text_config.hidden_size,), 7.0),
        "text_model.encoder.layers.0.layer_norm1.weight": _normal((text_config.hidden_size,), 4.0),
        "text_model.encoder.layers.0.layer_norm1.bias": _normal((text_config.hidden_size,), 6.0),
        "text_model.encoder.layers.0.mlp.fc1.weight": _normal(
            (text_config.intermediate_size, text_config.hidden_size), 4.5
        ),
        "text_model.encoder.layers.0.mlp.fc1.bias": _normal((text_config.intermediate_size,), 6.5),
        "text_model.encoder.layers.0.mlp.fc2.weight": _normal(
            (text_config.hidden_size, text_config.intermediate_size), 4.5
        ),
        "text_model.encoder.layers.0.mlp.fc2.bias": _normal((text_config.hidden_size,), 6.5),
        "text_model.encoder.layers.0.layer_norm2.weight": _normal((text_config.hidden_size,), 4.0),
        "text_model.encoder.layers.0.layer_norm2.bias": _normal((text_config.hidden_size,), 6.0),
        "text_model.final_layer_norm.weight": _normal((text_config.hidden_size,), 4.0),
        "text_model.final_layer_norm.bias": _normal((text_config.hidden_size,), 6.0),
        "text_projection.weight": _normal((text_config.projection_dim, text_config.hidden_size), 4.0),
        "vision_model.embeddings.class_embedding": _normal((vision_config.hidden_size,), 4.0),
        "vision_model.embeddings.patch_embedding.weight": _normal(
            (
                vision_config.hidden_size,
                vision_config.num_channels,
                vision_config.patch_size,
                vision_config.patch_size,
            ),
            5.0,
        ),
        "vision_model.embeddings.position_embedding.weight": _normal(
            (vision_config.num_positions, vision_config.hidden_size),
            4.5,
        ),
        "vision_model.pre_layrnorm.weight": _normal((vision_config.hidden_size,), 3.5),
        "vision_model.pre_layrnorm.bias": _normal((vision_config.hidden_size,), 5.5),
        "vision_model.post_layernorm.weight": _normal((vision_config.hidden_size,), 3.5),
        "vision_model.post_layernorm.bias": _normal((vision_config.hidden_size,), 5.5),
        "vision_model.encoder.layers.0.layer_norm1.weight": _normal((vision_config.hidden_size,), 4.0),
        "vision_model.encoder.layers.0.layer_norm1.bias": _normal((vision_config.hidden_size,), 6.0),
        "vision_model.encoder.layers.0.self_attn.q_proj.weight": _normal(
            (vision_config.hidden_size, vision_config.hidden_size), 5.0
        ),
        "vision_model.encoder.layers.0.self_attn.q_proj.bias": _normal((vision_config.hidden_size,), 7.0),
        "vision_model.encoder.layers.0.self_attn.k_proj.weight": _normal(
            (vision_config.hidden_size, vision_config.hidden_size), 5.0
        ),
        "vision_model.encoder.layers.0.self_attn.k_proj.bias": _normal((vision_config.hidden_size,), 7.0),
        "vision_model.encoder.layers.0.self_attn.v_proj.weight": _normal(
            (vision_config.hidden_size, vision_config.hidden_size), 5.0
        ),
        "vision_model.encoder.layers.0.self_attn.v_proj.bias": _normal((vision_config.hidden_size,), 7.0),
        "vision_model.encoder.layers.0.self_attn.out_proj.weight": _normal(
            (vision_config.hidden_size, vision_config.hidden_size), 5.0
        ),
        "vision_model.encoder.layers.0.self_attn.out_proj.bias": _normal((vision_config.hidden_size,), 7.0),
        "vision_model.encoder.layers.0.layer_norm2.weight": _normal((vision_config.hidden_size,), 4.0),
        "vision_model.encoder.layers.0.layer_norm2.bias": _normal((vision_config.hidden_size,), 6.0),
        "vision_model.encoder.layers.0.mlp.fc1.weight": _normal(
            (vision_config.intermediate_size, vision_config.hidden_size), 4.0
        ),
        "vision_model.encoder.layers.0.mlp.fc1.bias": _normal((vision_config.intermediate_size,), 6.0),
        "vision_model.encoder.layers.0.mlp.fc2.weight": _normal(
            (vision_config.hidden_size, vision_config.intermediate_size), 4.0
        ),
        "vision_model.encoder.layers.0.mlp.fc2.bias": _normal((vision_config.hidden_size,), 6.0),
        "visual_projection.weight": _normal((vision_config.projection_dim, vision_config.hidden_size), 4.0),
        "logit_scale": np.array(np.log(1.7), dtype=np.float32),
    }


WEIGHTS = build_weights()


def build_validation_inputs() -> dict[str, np.ndarray]:
    pixel_values = np.linspace(
        -1.5,
        1.5,
        num=IMAGE_BATCH * NUM_CHANNELS * IMAGE_SIZE * IMAGE_SIZE,
        dtype=np.float32,
    ).reshape(IMAGE_BATCH, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    return {
        "input_ids": np.array(
            [
                [0, 5, 15, 1],
                [0, 15, 4, 1],
            ],
            dtype=np.int64,
        ),
        "attention_mask": np.array(
            [
                [True, True, True, False],
                [True, True, True, False],
            ],
            dtype=np.bool_,
        ),
        "pixel_values": pixel_values,
    }


def build_spec() -> dml.ir.ModelSpec:
    return dml.trace(
        LegacyCLIPModel(build_text_config(), build_vision_config(), WEIGHTS),
        inputs={
            "input_ids": dml.TensorSpec([TEXT_BATCH, SEQ_LEN], "int64"),
            "pixel_values": dml.TensorSpec([IMAGE_BATCH, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE], "float32"),
            "attention_mask": dml.TensorSpec([TEXT_BATCH, SEQ_LEN], "bool"),
        },
        name="clip_model_workflow",
    )


class _CLIPTextFeaturesModule(dml.Module):
    def __init__(self):
        self.model = LegacyCLIPModel(build_text_config(), build_vision_config(), WEIGHTS)

    def forward(self, input_ids, attention_mask):
        return dml.ops.output(self.model.get_text_features(input_ids, attention_mask), "text_features")


class _CLIPImageFeaturesModule(dml.Module):
    def __init__(self):
        self.model = LegacyCLIPModel(build_text_config(), build_vision_config(), WEIGHTS)

    def forward(self, pixel_values):
        return dml.ops.output(self.model.get_image_features(pixel_values), "image_features")


def build_text_features_spec() -> dml.ir.ModelSpec:
    return dml.trace(
        _CLIPTextFeaturesModule(),
        inputs={
            "input_ids": dml.TensorSpec([TEXT_BATCH, SEQ_LEN], "int64"),
            "attention_mask": dml.TensorSpec([TEXT_BATCH, SEQ_LEN], "bool"),
        },
        name="clip_model_workflow_text_features",
    )


def build_image_features_spec() -> dml.ir.ModelSpec:
    return dml.trace(
        _CLIPImageFeaturesModule(),
        inputs={"pixel_values": dml.TensorSpec([IMAGE_BATCH, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE], "float32")},
        name="clip_model_workflow_image_features",
    )


def _import_local_transformers():
    if str(LOCAL_TRANSFORMERS_SRC) not in sys.path:
        sys.path.insert(0, str(LOCAL_TRANSFORMERS_SRC))
    transformers = __import__("transformers")
    resolved = Path(transformers.__file__).resolve()
    if not resolved.is_relative_to(LOCAL_TRANSFORMERS_SRC.resolve()):
        raise AssertionError(f"expected local /workspace/transformers import, got {resolved}")
    return transformers


def reference_outputs() -> dict[str, np.ndarray]:
    torch = __import__("torch")
    transformers = _import_local_transformers()

    text_config = transformers.CLIPTextConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=TEXT_HIDDEN,
        intermediate_size=TEXT_INTERMEDIATE,
        projection_dim=PROJECTION,
        num_attention_heads=NUM_HEADS,
        num_hidden_layers=1,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        hidden_act="quick_gelu",
        attention_dropout=0.0,
        layer_norm_eps=EPS,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
    )
    vision_config = transformers.CLIPVisionConfig(
        hidden_size=VISION_HIDDEN,
        intermediate_size=VISION_INTERMEDIATE,
        projection_dim=PROJECTION,
        num_attention_heads=NUM_HEADS,
        num_hidden_layers=1,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_channels=NUM_CHANNELS,
        hidden_act="quick_gelu",
        attention_dropout=0.0,
        layer_norm_eps=EPS,
    )
    config = transformers.CLIPConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        projection_dim=PROJECTION,
        logit_scale_init_value=float(np.asarray(WEIGHTS["logit_scale"], dtype=np.float32)),
    )
    model = transformers.CLIPModel(config)
    state_dict = model.state_dict()
    for name, value in WEIGHTS.items():
        if name == "logit_scale":
            state_dict[name] = torch.tensor(float(np.asarray(value, dtype=np.float32)), dtype=torch.float32)
        else:
            state_dict[name] = torch.from_numpy(np.asarray(value, dtype=np.float32))
    model.load_state_dict(state_dict)
    model.eval()

    inputs = build_validation_inputs()
    with torch.inference_mode():
        text_features = model.get_text_features(
            input_ids=torch.from_numpy(inputs["input_ids"]),
            attention_mask=torch.from_numpy(inputs["attention_mask"]),
        )
        image_features = model.get_image_features(pixel_values=torch.from_numpy(inputs["pixel_values"]))
        outputs = model(
            input_ids=torch.from_numpy(inputs["input_ids"]),
            attention_mask=torch.from_numpy(inputs["attention_mask"]),
            pixel_values=torch.from_numpy(inputs["pixel_values"]),
        )
    return {
        "text_features": text_features.pooler_output.detach().cpu().numpy().astype(np.float32),
        "image_features": image_features.pooler_output.detach().cpu().numpy().astype(np.float32),
        "logits_per_image": outputs.logits_per_image.detach().cpu().numpy().astype(np.float32),
        "logits_per_text": outputs.logits_per_text.detach().cpu().numpy().astype(np.float32),
        "text_embeds": outputs.text_embeds.detach().cpu().numpy().astype(np.float32),
        "image_embeds": outputs.image_embeds.detach().cpu().numpy().astype(np.float32),
    }


def inspect_workflow() -> dict[str, object]:
    spec = build_spec()
    node_op_counts = Counter(node["op"] for node in spec.ir["nodes"])
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    cuda_sources = collect_generated_sources("cuda", lowered["nodes"], tensor_map)
    manifest = build_kernel_manifest(lowered, CUDA_TARGET)
    required = manifest["required_kernels"]
    provider_ops = {"conv2d_bias", "gemm_rcr_bias", "gemm_rcr_bias_fast_gelu", "bmm_rcr", "bmm_rrr", "gemm_rcr"}
    provider_entries = [entry for entry in required if entry["op"] in provider_ops]
    model_entries = [entry for entry in required if entry["op"] not in provider_ops]
    return {
        "name": spec.name,
        "input_names": [tensor["name"] for tensor in spec.ir["inputs"]],
        "output_names": [tensor["name"] for tensor in spec.ir["outputs"]],
        "output_shapes": {tensor["name"]: tensor["shape"] for tensor in spec.ir["outputs"]},
        "node_op_counts": dict(sorted(node_op_counts.items())),
        "provider_kernel_ops": sorted({entry["op"] for entry in provider_entries}),
        "provider_kernel_libraries": sorted({entry["kernel_library"] for entry in provider_entries}),
        "required_kernel_libraries": sorted({entry["kernel_library"] for entry in required}),
        "generated_cuda_kernel_count": len(cuda_sources["kernels"]),
        "uses_traced_default_text_positions": "position_ids" not in {tensor["name"] for tensor in spec.ir["inputs"]},
        "vision_input_shape": [IMAGE_BATCH, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE],
        "patch_grid": [IMAGE_SIZE // PATCH_SIZE, IMAGE_SIZE // PATCH_SIZE],
        "has_cutlass_conv_scaffold": any(
            entry["op"] == "conv2d_bias"
            and entry["kernel_library"] == "cutlass_conv"
            and entry.get("cutlass_conv_plan", {}).get("status") == "manifest_scaffold_only"
            for entry in required
        ),
        "uses_similarity_transpose": node_op_counts.get("permute", 0) >= 1,
        "model_kernel_ops": sorted({entry["op"] for entry in model_entries}),
    }


def _round_array(value: np.ndarray) -> list[object]:
    return np.round(np.asarray(value, dtype=np.float32), 6).tolist()


def _tensor_parity(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    diff = np.abs(actual - expected)
    max_abs_diff = float(diff.max()) if diff.size else 0.0
    return {
        "allclose": bool(np.allclose(actual, expected, atol=ATOL, rtol=RTOL)),
        "max_abs_diff": max_abs_diff,
    }


def _output_parity(actual: dict[str, np.ndarray], expected: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return {name: _tensor_parity(actual[name], expected[name]) for name in MODEL_OUTPUT_NAMES}


def _run_compiled_cpu_artifact(
    artifact_dir: Path | None,
    *,
    spec: dml.ir.ModelSpec,
    inputs: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    temporary_dir: tempfile.TemporaryDirectory[str] | None = None
    retained = artifact_dir is not None
    if artifact_dir is None:
        temporary_dir = tempfile.TemporaryDirectory(prefix="clip_model_cpu_artifact_")
        artifact_dir = Path(temporary_dir.name) / "clip_model_workflow_cpu.dinoml"
    artifact_dir = Path(artifact_dir).resolve()

    try:
        artifact = dml.compile(spec, dml.Target("cpu"), artifact_dir)
        manifest_path = artifact.path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        generated_path = artifact.path / "debug" / "generated_src" / "module.cpp"
        generated = generated_path.read_text(encoding="utf-8")
        module = runtime.load(artifact.path)
        session = module.create_session()
        try:
            actual = session.run_numpy(inputs)
        finally:
            session.close()
            module.close()

        bridge_kernels = [
            name
            for name in ("conv2d_bias", "gemm_rcr", "gemm_rcr_bias", "gemm_rcr_bias_fast_gelu", "bmm_rcr", "bmm_rrr")
            if f"static int {name}_" in generated
        ]
        artifact_summary = {
            "path": str(artifact.path),
            "retained": retained,
            "module_exists": (artifact.path / "module.so").exists(),
            "manifest_exists": manifest_path.exists(),
            "generated_module_exists": generated_path.exists(),
            "target": manifest["target"],
            "bridge_kernels": bridge_kernels,
        }
        return actual, artifact_summary
    finally:
        if temporary_dir is not None:
            temporary_dir.cleanup()


def run_example(*, artifact_dir: str | Path | None = None) -> dict[str, object]:
    inputs = build_validation_inputs()
    spec = build_spec()
    eager_outputs = execute_cpu(spec, inputs)
    text_only = execute_cpu(
        build_text_features_spec(),
        {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
    )["text_features"]
    image_only = execute_cpu(build_image_features_spec(), {"pixel_values": inputs["pixel_values"]})["image_features"]
    transformers_outputs = reference_outputs()
    artifact_outputs, artifact_summary = _run_compiled_cpu_artifact(Path(artifact_dir) if artifact_dir is not None else None, spec=spec, inputs=inputs)
    summary = inspect_workflow()
    summary["inputs"] = {
        "input_ids": inputs["input_ids"].tolist(),
        "attention_mask": inputs["attention_mask"].tolist(),
        "pixel_values_shape": list(inputs["pixel_values"].shape),
    }
    summary["artifact"] = artifact_summary
    summary["feature_parity_vs_transformers"] = {
        "text_features": _tensor_parity(text_only, transformers_outputs["text_features"]),
        "image_features": _tensor_parity(image_only, transformers_outputs["image_features"]),
    }
    summary["parity"] = {
        "execute_cpu_vs_transformers": _output_parity(eager_outputs, transformers_outputs),
        "artifact_vs_execute_cpu": _output_parity(artifact_outputs, eager_outputs),
        "artifact_vs_transformers": _output_parity(artifact_outputs, transformers_outputs),
    }
    summary["text_features"] = _round_array(text_only)
    summary["image_features"] = _round_array(image_only)
    for name in MODEL_OUTPUT_NAMES:
        summary[name] = _round_array(artifact_outputs[name])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile and run a bounded CLIP CPU artifact workflow proof.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Optional output path for the compiled .dinoml artifact. Defaults to a temporary directory.",
    )
    args = parser.parse_args()
    print(json.dumps(run_example(artifact_dir=args.artifact_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
