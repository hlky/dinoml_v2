from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

import dinoml as dml
from dinoml.reference import reference_numpy
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.lowering.ops import collect_generated_sources
from dinoml.models.clip import LegacyCLIPTextConfig, LegacyCLIPTextModelWithProjection
from dinoml.passes import PassManager, validate_ir


BATCH = 2
SEQ_LEN = 4
VOCAB_SIZE = 16
HIDDEN = 6
NUM_HEADS = 2
INTERMEDIATE = 8
PROJECTION = 5
EPS = 1.0e-5
MAX_POSITION_EMBEDDINGS = 6
CUDA_TARGET = {"name": "cuda", "arch": "sm_86"}


def build_config(*, eos_token_id: int = 2) -> LegacyCLIPTextConfig:
    return LegacyCLIPTextConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_attention_heads=NUM_HEADS,
        num_hidden_layers=1,
        projection_dim=PROJECTION,
        layer_norm_eps=EPS,
        eos_token_id=eos_token_id,
    )


def build_weights() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(2031)

    def _normal(shape, scale):
        return (rng.standard_normal(shape).astype(np.float32) / scale).astype(np.float32)

    config = build_config()
    return {
        "text_model.embeddings.token_embedding.weight": _normal((config.vocab_size, config.hidden_size), 3.5),
        "text_model.embeddings.position_embedding.weight": _normal(
            (config.max_position_embeddings, config.hidden_size), 4.0
        ),
        "text_model.encoder.layers.0.self_attn.q_proj.weight": _normal((config.hidden_size, config.hidden_size), 5.0),
        "text_model.encoder.layers.0.self_attn.q_proj.bias": _normal((config.hidden_size,), 7.0),
        "text_model.encoder.layers.0.self_attn.k_proj.weight": _normal((config.hidden_size, config.hidden_size), 5.0),
        "text_model.encoder.layers.0.self_attn.k_proj.bias": _normal((config.hidden_size,), 7.0),
        "text_model.encoder.layers.0.self_attn.v_proj.weight": _normal((config.hidden_size, config.hidden_size), 5.0),
        "text_model.encoder.layers.0.self_attn.v_proj.bias": _normal((config.hidden_size,), 7.0),
        "text_model.encoder.layers.0.self_attn.out_proj.weight": _normal(
            (config.hidden_size, config.hidden_size), 5.0
        ),
        "text_model.encoder.layers.0.self_attn.out_proj.bias": _normal((config.hidden_size,), 7.0),
        "text_model.encoder.layers.0.layer_norm1.weight": _normal((config.hidden_size,), 4.0),
        "text_model.encoder.layers.0.layer_norm1.bias": _normal((config.hidden_size,), 6.0),
        "text_model.encoder.layers.0.mlp.fc1.weight": _normal((config.intermediate_size, config.hidden_size), 4.5),
        "text_model.encoder.layers.0.mlp.fc1.bias": _normal((config.intermediate_size,), 6.5),
        "text_model.encoder.layers.0.mlp.fc2.weight": _normal((config.hidden_size, config.intermediate_size), 4.5),
        "text_model.encoder.layers.0.mlp.fc2.bias": _normal((config.hidden_size,), 6.5),
        "text_model.encoder.layers.0.layer_norm2.weight": _normal((config.hidden_size,), 4.0),
        "text_model.encoder.layers.0.layer_norm2.bias": _normal((config.hidden_size,), 6.0),
        "text_model.final_layer_norm.weight": _normal((config.hidden_size,), 4.0),
        "text_model.final_layer_norm.bias": _normal((config.hidden_size,), 6.0),
        "text_projection.weight": _normal((config.projection_dim, config.hidden_size), 4.0),
    }


WEIGHTS = build_weights()


def build_validation_inputs(*, eos_token_id: int = 2) -> dict[str, np.ndarray]:
    if eos_token_id == 2:
        input_ids = np.array(
            [
                [0, 5, 15, 1],
                [0, 15, 4, 1],
            ],
            dtype=np.int64,
        )
    else:
        input_ids = np.array(
            [
                [0, eos_token_id, 3, eos_token_id],
                [5, 4, eos_token_id, eos_token_id],
            ],
            dtype=np.int64,
        )
    return {
        "input_ids": input_ids,
        "attention_mask": np.array(
            [
                [True, True, True, False],
                [True, True, True, False],
            ],
            dtype=np.bool_,
        ),
    }


def build_spec(*, eos_token_id: int = 2) -> dml.ir.ModelSpec:
    return dml.trace(
        LegacyCLIPTextModelWithProjection(build_config(eos_token_id=eos_token_id), WEIGHTS),
        inputs={
            "input_ids": dml.TensorSpec([BATCH, SEQ_LEN], "int64"),
            "attention_mask": dml.TensorSpec([BATCH, SEQ_LEN], "bool"),
        },
        name=f"clip_text_workflow_eos_{eos_token_id}",
    )


def reference_outputs(*, eos_token_id: int = 2) -> np.ndarray:
    torch = __import__("torch")
    transformers = __import__("transformers")

    text_config = transformers.CLIPTextConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        projection_dim=PROJECTION,
        num_attention_heads=NUM_HEADS,
        num_hidden_layers=1,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        hidden_act="quick_gelu",
        attention_dropout=0.0,
        layer_norm_eps=EPS,
        bos_token_id=0,
        eos_token_id=eos_token_id,
        pad_token_id=1 if eos_token_id == 2 else eos_token_id,
    )
    vision_config = transformers.CLIPVisionConfig(
        hidden_size=8,
        intermediate_size=16,
        projection_dim=PROJECTION,
        num_attention_heads=2,
        num_hidden_layers=1,
        image_size=4,
        patch_size=2,
    )
    config = transformers.CLIPConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        projection_dim=PROJECTION,
    )
    model = transformers.CLIPModel(config)
    state_dict = model.state_dict()
    for name, value in WEIGHTS.items():
        state_dict[name] = torch.from_numpy(np.asarray(value, dtype=np.float32))
    model.load_state_dict(state_dict)
    model.eval()

    inputs = build_validation_inputs(eos_token_id=eos_token_id)
    kwargs = {
        "input_ids": torch.from_numpy(inputs["input_ids"]),
        "attention_mask": torch.from_numpy(inputs["attention_mask"]),
    }
    with torch.inference_mode():
        text_features = model.get_text_features(**kwargs)
    if hasattr(text_features, "pooler_output"):
        text_features = text_features.pooler_output
    return text_features.detach().cpu().numpy().astype(np.float32)


def inspect_workflow(*, eos_token_id: int = 2) -> dict[str, object]:
    spec = build_spec(eos_token_id=eos_token_id)
    node_op_counts = Counter(node["op"] for node in spec.ir["nodes"])
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    cuda_sources = collect_generated_sources("cuda", lowered["nodes"], tensor_map)
    manifest = build_kernel_manifest(lowered, CUDA_TARGET)
    required = manifest["required_kernels"]
    provider_ops = {"gemm_rcr_bias", "gemm_rcr_bias_quick_gelu", "gemm_rcr", "bmm_rcr", "bmm_rrr"}
    provider_entries = [entry for entry in required if entry["op"] in provider_ops]
    model_entries = [entry for entry in required if entry["op"] not in provider_ops]
    return {
        "name": spec.name,
        "eos_token_id": eos_token_id,
        "output_name": spec.ir["outputs"][0]["name"],
        "output_shape": spec.ir["outputs"][0]["shape"],
        "node_op_counts": dict(sorted(node_op_counts.items())),
        "provider_kernel_ops": sorted({entry["op"] for entry in provider_entries}),
        "provider_kernel_libraries": sorted({entry["kernel_library"] for entry in provider_entries}),
        "model_kernel_ops": sorted({entry["op"] for entry in model_entries}),
        "required_kernel_libraries": sorted({entry["kernel_library"] for entry in required}),
        "generated_cuda_kernel_count": len(cuda_sources["kernels"]),
        "has_integer_eos_eq_kernel": any("dinoml::math::eq(" in source for source in cuda_sources["kernels"]),
        "uses_batch_gather_pooling": node_op_counts["batch_gather"] == 1,
    }


def run_example(*, eos_token_id: int = 2) -> dict[str, object]:
    inputs = build_validation_inputs(eos_token_id=eos_token_id)
    actual = reference_numpy(build_spec(eos_token_id=eos_token_id), inputs)["text_features"]
    summary = inspect_workflow(eos_token_id=eos_token_id)
    summary["inputs"] = {
        "input_ids": inputs["input_ids"].tolist(),
        "attention_mask": inputs["attention_mask"].tolist(),
    }
    summary["text_features"] = np.round(actual, 6).tolist()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the bounded CLIP text workflow proof.")
    parser.add_argument("--eos-token-id", type=int, default=2, help="Trace the wrapper with this EOS token id.")
    args = parser.parse_args()
    print(json.dumps(run_example(eos_token_id=args.eos_token_id), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
