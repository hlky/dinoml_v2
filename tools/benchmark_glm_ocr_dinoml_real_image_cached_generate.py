from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Mapping
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

import dinoml as dml
from dinoml import runtime
from dinoml.ir import array_to_storage
from dinoml.models.glm_ocr import (
    GlmOcrForConditionalGenerationDecodeSessionStaticCache,
    GlmOcrForConditionalGenerationDecodeStaticCache,
    GlmOcrForConditionalGenerationImagePrefillWithCache,
    glm_ocr_config_from_transformers_dict,
    glm_ocr_required_text_weight_names,
    glm_ocr_rope_index,
    glm_ocr_text_rope_embeddings,
    glm_ocr_vision_position_ids,
    glm_ocr_vision_rope_embeddings,
    glm_ocr_weights_from_safetensors_file,
)
from dinoml.models.kv_cache import (
    StaticKvCacheSpec,
    empty_static_kv_cache,
    seed_static_kv_cache,
    static_kv_cache_input_specs,
    write_static_kv_cache_update,
)
try:
    from tools.glm_ocr_benchmark_common import (
        DEFAULT_IMAGE,
        DEFAULT_PROMPT,
        DEFAULT_SNAPSHOT,
        configure_processor_image_size,
        open_rgb_image,
    )
except ModuleNotFoundError:
    from glm_ocr_benchmark_common import (
        DEFAULT_IMAGE,
        DEFAULT_PROMPT,
        DEFAULT_SNAPSHOT,
        configure_processor_image_size,
        open_rgb_image,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark real-image GLM-OCR generation with DinoML KV cache.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--benchmark-warmup", type=int, default=5)
    parser.add_argument("--benchmark-iterations", type=int, default=20)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument(
        "--attention-mask",
        action="store_true",
        help="Compile and run additive prefill/decode attention masks; ROCm flash paths use CK bias kernels.",
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--target", choices=("rocm", "cuda", "cpu"), default="rocm")
    parser.add_argument("--arch", default=None)
    parser.add_argument("--longest-side", type=int, default=None, help="Resize the source image to this longest edge before processing.")
    parser.add_argument("--min-pixels", type=int, default=None, help="Override processor shortest_edge pixel-count bound.")
    parser.add_argument("--max-pixels", type=int, default=None, help="Override processor longest_edge pixel-count bound.")
    parser.add_argument("--prefill-artifact", type=Path)
    parser.add_argument("--decode-artifact", type=Path)
    parser.add_argument("--prefill-execution-plan", type=Path)
    parser.add_argument("--decode-execution-plan", type=Path)
    parser.add_argument("--profile-compile", action="store_true")
    parser.add_argument("--profile-iterations", type=int, default=5)
    parser.add_argument("--profile-repeats", type=int, default=1)
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def build_config(snapshot: Path, dtype: str):
    payload = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    return glm_ocr_config_from_transformers_dict(payload, dtype=dtype)


def processor_inputs(processor, image_path: Path, prompt: str, *, longest_side: int | None = None):
    image, source_image_size = open_rgb_image(image_path, longest_side=longest_side)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"return_mm_token_type_ids": True},
        )
    except TypeError:
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text], images=[image], return_tensors="pt", return_mm_token_type_ids=True)
    return {name: value.detach().cpu().numpy() for name, value in inputs.items()}, image.size, source_image_size


def build_inputs(config, processed: dict[str, np.ndarray], max_new_tokens: int, *, use_attention_mask: bool = False):
    input_ids = np.asarray(processed["input_ids"], dtype=np.int64)
    mm_token_type_ids = np.asarray(processed["mm_token_type_ids"], dtype=np.int64)
    image_grid_thw = np.asarray(processed["image_grid_thw"], dtype=np.int64)
    pixel_values = np.asarray(processed["pixel_values"], dtype=np.float32)
    prompt_len = int(input_ids.shape[1])
    max_cache_len = prompt_len + max_new_tokens

    decode_ids = np.full((1, max_cache_len), 0, dtype=np.int64)
    decode_ids[:, :prompt_len] = input_ids
    decode_mm = np.zeros((1, max_cache_len), dtype=np.int64)
    decode_mm[:, :prompt_len] = mm_token_type_ids
    text_position_ids, _ = glm_ocr_rope_index(
        decode_ids,
        decode_mm,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=config.vision_config.spatial_merge_size,
    )
    text_cos, text_sin = glm_ocr_text_rope_embeddings(text_position_ids, config.text_config, dtype=config.text_config.dtype)
    vision_position_ids = glm_ocr_vision_position_ids(image_grid_thw, config.vision_config.spatial_merge_size)
    vision_cos, vision_sin = glm_ocr_vision_rope_embeddings(
        vision_position_ids,
        head_dim=config.vision_config.head_dim,
    )
    image_positions = np.flatnonzero(mm_token_type_ids[0] == 1)
    if image_positions.size == 0:
        image_positions = np.flatnonzero(input_ids[0] == config.image_token_id)
    if image_positions.size == 0:
        raise RuntimeError("processor inputs did not contain image placeholder tokens")
    inputs = {
        "input_ids": input_ids,
        "pixel_values": _float_input(pixel_values, config.vision_config.dtype),
        "vision_cos": _float_input(vision_cos, "float32"),
        "vision_sin": _float_input(vision_sin, "float32"),
        "text_cos": _float_input(text_cos, config.text_config.dtype),
        "text_sin": _float_input(text_sin, config.text_config.dtype),
    }
    if use_attention_mask:
        inputs["attention_mask"] = _prefill_attention_mask(
            config,
            processed.get("attention_mask"),
            prompt_len=prompt_len,
        )
    return inputs, prompt_len, max_cache_len, int(image_positions[0])


def build_prefill_spec(config, weights: dict[str, np.ndarray], inputs: dict[str, np.ndarray], image_token_start: int):
    return dml.trace(
        GlmOcrForConditionalGenerationImagePrefillWithCache(
            config,
            weights,
            image_token_start=image_token_start,
            logits_to_keep=1,
            vision_cos=inputs["vision_cos"],
            vision_sin=inputs["vision_sin"],
            text_cos=inputs["text_cos"][:, : inputs["input_ids"].shape[1], :],
            text_sin=inputs["text_sin"][:, : inputs["input_ids"].shape[1], :],
        ),
        inputs={
            "input_ids": dml.TensorSpec(list(inputs["input_ids"].shape), "int64"),
            "pixel_values": dml.TensorSpec(list(inputs["pixel_values"].shape), config.vision_config.dtype),
            **(
                {
                    "attention_mask": dml.TensorSpec(
                        list(inputs["attention_mask"].shape),
                        config.text_config.dtype,
                    )
                }
                if "attention_mask" in inputs
                else {}
            ),
        },
        name=f"glm_ocr_real_image_cached_prefill_s{inputs['input_ids'].shape[1]}_p{inputs['pixel_values'].shape[0]}",
    )


def build_decode_spec(
    config,
    weights: dict[str, np.ndarray],
    max_cache_len: int,
    *,
    use_flash_static_kv_cache: bool,
    use_session_static_kv_cache: bool,
    use_decode_attention_mask: bool = False,
):
    cache_spec = cache_spec_for_config(config, max_cache_len)
    inputs = {
        "input_ids": dml.TensorSpec([1, 1], "int64"),
        "cos": dml.TensorSpec([1, 1, config.text_config.head_dim], config.text_config.dtype),
        "sin": dml.TensorSpec([1, 1, config.text_config.head_dim], config.text_config.dtype),
    }
    if use_decode_attention_mask or not use_flash_static_kv_cache:
        mask_cache_len = max_cache_len if use_flash_static_kv_cache else max_cache_len + 1
        inputs["attention_mask"] = dml.TensorSpec(
            [config.text_config.num_attention_heads, 1, mask_cache_len],
            config.text_config.dtype,
        )
    if use_flash_static_kv_cache and not use_session_static_kv_cache:
        inputs["cache_seqlens"] = dml.TensorSpec([1], "int32")
    if not use_session_static_kv_cache:
        inputs.update(static_kv_cache_input_specs(cache_spec))
        model = GlmOcrForConditionalGenerationDecodeStaticCache(config, weights)
    else:
        model = GlmOcrForConditionalGenerationDecodeSessionStaticCache(
            config,
            weights,
            max_cache_len=max_cache_len,
        )
    return dml.trace(
        model,
        inputs=inputs,
        name=f"glm_ocr_real_image_cached_decode_past{max_cache_len}",
    )


def cache_spec_for_config(config, max_cache_len: int) -> StaticKvCacheSpec:
    return StaticKvCacheSpec(
        num_layers=config.text_config.num_hidden_layers,
        batch=1,
        num_key_value_heads=config.text_config.num_key_value_heads,
        max_cache_len=max_cache_len,
        head_dim=config.text_config.head_dim,
        dtype=config.text_config.dtype,
    )


def ensure_artifacts(args: argparse.Namespace, config, inputs, image_token_start: int, max_cache_len: int) -> tuple[Path, Path]:
    use_flash_static_kv_cache = _use_flash_static_kv_cache(args, config)
    use_session_static_kv_cache = _use_session_static_kv_cache(args, config)
    use_decode_attention_mask = bool(getattr(args, "attention_mask", False))
    flash_suffix = "_flash" if getattr(config.text_config, "use_flash_attention", True) else ""
    mask_suffix = "_mask" if bool(getattr(args, "attention_mask", False)) else ""
    rope_suffix = "_rope_const"
    prefill_artifact = args.prefill_artifact
    if prefill_artifact is None:
        prefill_artifact = Path("build") / (
            f"glm_ocr_real_image_cached_prefill_s{inputs['input_ids'].shape[1]}"
            f"_p{inputs['pixel_values'].shape[0]}{rope_suffix}{mask_suffix}{flash_suffix}_{args.dtype}_{args.target}.dinoml"
        )
    decode_artifact = args.decode_artifact
    if decode_artifact is None:
        if use_session_static_kv_cache:
            cache_suffix = "_session_static_kv"
        else:
            cache_suffix = "_flash_static_kv" if use_flash_static_kv_cache else ""
        decode_artifact = Path("build") / (
            f"glm_ocr_real_image_cached_decode_past{max_cache_len}{cache_suffix}{mask_suffix}{flash_suffix}_{args.dtype}_{args.target}.dinoml"
        )
    prefill_ready = _prefill_artifact_compatible(prefill_artifact, config, expected_inputs=inputs)
    decode_ready = _decode_artifact_compatible(
        decode_artifact,
        config,
        max_cache_len,
        use_flash_static_kv_cache=use_flash_static_kv_cache,
        use_session_static_kv_cache=use_session_static_kv_cache,
        use_decode_attention_mask=use_decode_attention_mask,
    )
    if not args.force_compile:
        if args.prefill_artifact is not None and _artifact_ready(prefill_artifact) and not prefill_ready:
            raise ValueError(
                f"Prefill artifact {prefill_artifact} is not compatible with the requested GLM-OCR benchmark. "
                "Pass a compatible artifact or use --force-compile to overwrite it."
            )
        if args.decode_artifact is not None and _artifact_ready(decode_artifact) and not decode_ready:
            raise ValueError(
                f"Decode artifact {decode_artifact} is not compatible with the requested GLM-OCR benchmark. "
                "Pass a compatible artifact or use --force-compile to overwrite it."
            )
    if prefill_ready and decode_ready and not args.force_compile:
        return prefill_artifact, decode_artifact

    if args.force_compile or not prefill_ready:
        full_weights = glm_ocr_weights_from_safetensors_file(
            args.snapshot / "model.safetensors",
            config,
            dtype=config.text_config.dtype,
        )
        prefill_artifact.parent.mkdir(parents=True, exist_ok=True)
        dml.compile(
            build_prefill_spec(config, full_weights, inputs, image_token_start),
            target=dml.Target(args.target, arch=args.arch),
            output=prefill_artifact,
            execution_plan=args.prefill_execution_plan,
            profile=args.profile_compile,
            profile_iterations=args.profile_iterations,
            profile_repeats=args.profile_repeats,
        )
    if args.force_compile or not decode_ready:
        text_weights = glm_ocr_weights_from_safetensors_file(
            args.snapshot / "model.safetensors",
            config,
            dtype=config.text_config.dtype,
            required_names=glm_ocr_required_text_weight_names(config),
        )
        decode_artifact.parent.mkdir(parents=True, exist_ok=True)
        dml.compile(
            build_decode_spec(
                config,
                text_weights,
                max_cache_len,
                use_flash_static_kv_cache=use_flash_static_kv_cache,
                use_session_static_kv_cache=use_session_static_kv_cache,
                use_decode_attention_mask=use_decode_attention_mask,
            ),
            target=dml.Target(args.target, arch=args.arch),
            output=decode_artifact,
            execution_plan=args.decode_execution_plan,
            profile=args.profile_compile,
            profile_iterations=args.profile_iterations,
            profile_repeats=args.profile_repeats,
        )
    return prefill_artifact, decode_artifact


def _artifact_ready(path: Path) -> bool:
    return (path / "manifest.json").is_file() and (path / "module.so").is_file()


def _prefill_artifact_compatible(path: Path, config, *, expected_inputs: Mapping[str, np.ndarray] | None = None) -> bool:
    metadata = _artifact_metadata(path)
    if metadata is None:
        return False
    input_names = _metadata_names(metadata, "inputs")
    output_names = _metadata_names(metadata, "outputs")
    uses_attention_mask = expected_inputs is not None and "attention_mask" in expected_inputs
    required_inputs = {"input_ids", "pixel_values"}
    if uses_attention_mask:
        required_inputs.add("attention_mask")
    required_outputs = {"logits"} | {
        name
        for layer_idx in range(config.text_config.num_hidden_layers)
        for name in (f"present_key_{layer_idx}", f"present_value_{layer_idx}")
    }
    if input_names != required_inputs or not required_outputs <= output_names:
        return False
    kernel_ops = _artifact_kernel_ops(path)
    required_flash_op = "flash_attention_bias" if uses_attention_mask else "flash_attention"
    if required_flash_op not in kernel_ops:
        return False
    graph_op_counts = _artifact_graph_op_counts(path)
    if graph_op_counts.get("swiglu") != int(config.text_config.num_hidden_layers):
        return False
    if expected_inputs is not None:
        prompt_len = int(expected_inputs["input_ids"].shape[1])
        input_expectations = {
            "input_ids": (list(expected_inputs["input_ids"].shape), "int64"),
            "pixel_values": (list(expected_inputs["pixel_values"].shape), config.vision_config.dtype),
        }
        if uses_attention_mask:
            input_expectations["attention_mask"] = (
                [config.text_config.num_attention_heads, prompt_len, prompt_len],
                config.text_config.dtype,
            )
        output_expectations = {
            "logits": ([1, 1, config.text_config.vocab_size], config.text_config.dtype),
            **{
                name: ([1, config.text_config.num_key_value_heads, prompt_len, config.text_config.head_dim], config.text_config.dtype)
                for layer_idx in range(config.text_config.num_hidden_layers)
                for name in (f"present_key_{layer_idx}", f"present_value_{layer_idx}")
            },
        }
        if not _metadata_tensors_match(metadata, "inputs", input_expectations):
            return False
        if not _metadata_tensors_match(metadata, "outputs", output_expectations):
            return False
    return True


def _decode_artifact_compatible(
    path: Path,
    config,
    max_cache_len: int,
    *,
    use_flash_static_kv_cache: bool,
    use_session_static_kv_cache: bool,
    use_decode_attention_mask: bool = False,
) -> bool:
    metadata = _artifact_metadata(path)
    if metadata is None:
        return False
    input_names = _metadata_names(metadata, "inputs")
    output_names = _metadata_names(metadata, "outputs")
    state_names = _metadata_names(metadata, "states")
    cache_spec = cache_spec_for_config(config, max_cache_len)
    state_cache_names = {
        name
        for layer_idx in range(cache_spec.num_layers)
        for name in (cache_spec.past_key_name(layer_idx), cache_spec.past_value_name(layer_idx))
    }
    input_cache_names = set(state_cache_names)
    output_update_names = {
        name
        for layer_idx in range(cache_spec.num_layers)
        for name in (cache_spec.new_key_name(layer_idx), cache_spec.new_value_name(layer_idx))
    }
    required_inputs = {"input_ids", "cos", "sin"}
    if use_decode_attention_mask or not use_flash_static_kv_cache:
        required_inputs.add("attention_mask")
    if use_flash_static_kv_cache and not use_session_static_kv_cache:
        required_inputs.add("cache_seqlens")
    if input_names != required_inputs:
        return False
    input_expectations = {
        "input_ids": ([1, 1], "int64"),
        "cos": ([1, 1, config.text_config.head_dim], config.text_config.dtype),
        "sin": ([1, 1, config.text_config.head_dim], config.text_config.dtype),
    }
    if use_decode_attention_mask or not use_flash_static_kv_cache:
        mask_cache_len = max_cache_len if use_flash_static_kv_cache else max_cache_len + 1
        input_expectations["attention_mask"] = (
            [config.text_config.num_attention_heads, 1, mask_cache_len],
            config.text_config.dtype,
        )
    if use_flash_static_kv_cache and not use_session_static_kv_cache:
        input_expectations["cache_seqlens"] = ([1], "int32")
    if not _metadata_tensors_match(metadata, "inputs", input_expectations):
        return False
    if use_session_static_kv_cache:
        expected_state_names = state_cache_names | {"cache_seqlens"}
        if state_names != expected_state_names:
            return False
        if input_cache_names & input_names:
            return False
        if output_names != {"logits"}:
            return False
        if not _metadata_tensors_match(
            metadata,
            "outputs",
            {"logits": ([1, 1, config.text_config.vocab_size], config.text_config.dtype)},
        ):
            return False
        if not _decode_states_have_expected_shape(metadata, cache_spec, include_cache_seqlens=True):
            return False
    else:
        if state_names:
            return False
        if not input_cache_names <= input_names:
            return False
        if not ({"logits"} | output_update_names) <= output_names:
            return False
        cache_expectations = {
            name: (cache_spec.shape(), cache_spec.dtype)
            for layer_idx in range(cache_spec.num_layers)
            for name in (cache_spec.past_key_name(layer_idx), cache_spec.past_value_name(layer_idx))
        }
        update_expectations = {
            name: (cache_spec.shape(cache_len=1), cache_spec.dtype)
            for layer_idx in range(cache_spec.num_layers)
            for name in (cache_spec.new_key_name(layer_idx), cache_spec.new_value_name(layer_idx))
        }
        if not _metadata_tensors_match(metadata, "inputs", cache_expectations):
            return False
        if not _metadata_tensors_match(
            metadata,
            "outputs",
            {"logits": ([1, 1, config.text_config.vocab_size], config.text_config.dtype), **update_expectations},
        ):
            return False
    if use_flash_static_kv_cache:
        required_op = "flash_attention_static_kv_cache_bias" if use_decode_attention_mask else "flash_attention_static_kv_cache"
        if required_op not in _artifact_kernel_ops(path):
            return False
    return True


def _artifact_metadata(path: Path) -> Mapping[str, Any] | None:
    if not _artifact_ready(path):
        return None
    metadata = _read_artifact_json(path, "metadata.json")
    return metadata if isinstance(metadata, Mapping) else None


def _artifact_kernel_ops(path: Path) -> set[str]:
    manifest = _read_artifact_json(path, "kernel_manifest.json")
    if not isinstance(manifest, Mapping):
        return set()
    return {
        str(item.get("op"))
        for item in manifest.get("required_kernels", [])
        if isinstance(item, Mapping) and item.get("op") is not None
    }


def _artifact_graph_op_counts(path: Path) -> dict[str, int]:
    graph = _read_artifact_json(path, "graph.dinoir.json")
    if not isinstance(graph, Mapping):
        return {}
    counts: dict[str, int] = {}
    nodes = graph.get("nodes", [])
    if not isinstance(nodes, list):
        return counts
    for node in nodes:
        if not isinstance(node, Mapping) or node.get("op") is None:
            continue
        op = str(node["op"])
        counts[op] = counts.get(op, 0) + 1
    return counts


def _read_artifact_json(path: Path, filename: str) -> object | None:
    try:
        return json.loads((path / filename).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _metadata_names(metadata: Mapping[str, Any], key: str) -> set[str]:
    items = metadata.get(key, [])
    if not isinstance(items, list):
        return set()
    return {str(item.get("name")) for item in items if isinstance(item, Mapping) and item.get("name") is not None}


def _metadata_tensors_match(
    metadata: Mapping[str, Any],
    key: str,
    expectations: Mapping[str, tuple[list[int], str]],
) -> bool:
    items = metadata.get(key, [])
    if not isinstance(items, list):
        return False
    by_name = {str(item.get("name")): item for item in items if isinstance(item, Mapping) and item.get("name") is not None}
    for name, (expected_shape, expected_dtype) in expectations.items():
        item = by_name.get(name)
        if item is None:
            return False
        try:
            actual_shape = [int(dim) for dim in item.get("shape", [])]
        except (TypeError, ValueError):
            return False
        if actual_shape != [int(dim) for dim in expected_shape]:
            return False
        if str(item.get("dtype")) != str(expected_dtype):
            return False
    return True


def _decode_states_have_expected_shape(
    metadata: Mapping[str, Any],
    spec: StaticKvCacheSpec,
    *,
    include_cache_seqlens: bool = False,
) -> bool:
    states = metadata.get("states", [])
    if not isinstance(states, list):
        return False
    expected_shape = spec.shape()
    expected_dtype = spec.dtype
    expected_names = {
        name
        for layer_idx in range(spec.num_layers)
        for name in (spec.past_key_name(layer_idx), spec.past_value_name(layer_idx))
    }
    if include_cache_seqlens:
        expected_names.add("cache_seqlens")
    by_name = {str(item.get("name")): item for item in states if isinstance(item, Mapping)}
    for name in expected_names:
        item = by_name.get(name)
        if item is None:
            return False
        if name == "cache_seqlens":
            if [int(dim) for dim in item.get("shape", [])] != [spec.batch]:
                return False
            if str(item.get("dtype")) != "int32":
                return False
        else:
            if [int(dim) for dim in item.get("shape", [])] != expected_shape:
                return False
            if str(item.get("dtype")) != expected_dtype:
                return False
    return True


def generate_once(
    prefill_session,
    decode_session,
    prefill_inputs: dict[str, np.ndarray],
    config,
    prompt_len: int,
    max_cache_len: int,
    max_new_tokens: int,
    eos_token_id: int,
    use_flash_static_kv_cache: bool = False,
    use_session_static_kv_cache: bool = False,
    use_decode_attention_mask: bool = False,
) -> tuple[list[int], list[float]]:
    cache_spec = cache_spec_for_config(config, max_cache_len)
    started = time.perf_counter()
    prefill_outputs, cache = _run_prefill_for_decode(
        prefill_session,
        decode_session,
        _prefill_run_inputs(prefill_inputs, prompt_len),
        cache_spec,
        cache_len=prompt_len,
        use_session_static_kv_cache=use_session_static_kv_cache,
    )
    run_times_ms = [(time.perf_counter() - started) * 1000.0]
    next_id = int(np.argmax(prefill_outputs["logits"][0, 0, :]))
    generated: list[int] = []

    for step in range(max_new_tokens):
        generated.append(next_id)
        if next_id == eos_token_id or step == max_new_tokens - 1:
            break
        position = prompt_len + step
        decode_inputs = {
            "input_ids": np.asarray([[next_id]], dtype=np.int64),
            "cos": _slice_position(prefill_inputs["text_cos"], position),
            "sin": _slice_position(prefill_inputs["text_sin"], position),
        }
        if use_decode_attention_mask or not use_flash_static_kv_cache:
            decode_inputs["attention_mask"] = _decode_attention_mask(
                config,
                max_cache_len,
                valid_past_len=position,
                use_flash_static_kv_cache=use_flash_static_kv_cache,
            )
        if not use_session_static_kv_cache:
            decode_inputs.update(cache)
        if use_flash_static_kv_cache and not use_session_static_kv_cache:
            decode_inputs["cache_seqlens"] = np.asarray([position], dtype=np.int32)
        started = time.perf_counter()
        decode_outputs = decode_session.run_numpy(decode_inputs)
        run_times_ms.append((time.perf_counter() - started) * 1000.0)
        if not use_session_static_kv_cache:
            write_static_kv_cache_update(cache, decode_outputs, cache_spec, position=position)
        next_id = int(np.argmax(decode_outputs["logits"][0, 0, :]))
    return generated, run_times_ms


def benchmark_modules_native(
    prefill_session,
    decode_session,
    prefill_inputs: dict[str, np.ndarray],
    config,
    prompt_len: int,
    max_cache_len: int,
    max_new_tokens: int,
    *,
    warmup: int,
    iterations: int,
    use_flash_static_kv_cache: bool,
    use_session_static_kv_cache: bool,
    use_decode_attention_mask: bool = False,
) -> dict[str, object]:
    prefill_run_inputs = _prefill_run_inputs(prefill_inputs, prompt_len)
    prefill_summary = prefill_session.benchmark_numpy(prefill_run_inputs, warmup=warmup, iterations=iterations)

    cache_spec = cache_spec_for_config(config, max_cache_len)
    prefill_outputs, cache = _run_prefill_for_decode(
        prefill_session,
        decode_session,
        prefill_run_inputs,
        cache_spec,
        cache_len=prompt_len,
        use_session_static_kv_cache=use_session_static_kv_cache,
    )
    next_id = int(np.argmax(prefill_outputs["logits"][0, 0, :]))
    first_decode_position = prompt_len
    late_decode_position = min(max_cache_len - 1, prompt_len + max(0, max_new_tokens - 2))

    first_decode_inputs = _decode_run_inputs(
        prefill_inputs,
        config,
        max_cache_len,
        next_id=next_id,
        position=first_decode_position,
        cache=cache,
        use_flash_static_kv_cache=use_flash_static_kv_cache,
        use_session_static_kv_cache=use_session_static_kv_cache,
        use_decode_attention_mask=use_decode_attention_mask,
    )
    first_decode_summary = decode_session.benchmark_numpy(
        first_decode_inputs,
        warmup=warmup,
        iterations=iterations,
    )

    late_cache = _cache_with_zero_filled_future(cache_spec) if not use_session_static_kv_cache else {}
    for name, value in cache.items():
        late_cache[name][...] = value
    if use_session_static_kv_cache:
        decode_session.set_state_numpy("cache_seqlens", np.asarray([late_decode_position], dtype=np.int32))
    late_decode_inputs = _decode_run_inputs(
        prefill_inputs,
        config,
        max_cache_len,
        next_id=next_id,
        position=late_decode_position,
        cache=late_cache,
        use_flash_static_kv_cache=use_flash_static_kv_cache,
        use_session_static_kv_cache=use_session_static_kv_cache,
        use_decode_attention_mask=use_decode_attention_mask,
    )
    late_decode_summary = decode_session.benchmark_numpy(
        late_decode_inputs,
        warmup=warmup,
        iterations=iterations,
    )

    prefill_median_ms = float(prefill_summary["median_ms"])
    first_decode_median_ms = float(first_decode_summary["median_ms"])
    late_decode_median_ms = float(late_decode_summary["median_ms"])
    decode_steps_for_max_new_tokens = max(0, int(max_new_tokens) - 1)
    return {
        "method": "session.benchmark_numpy",
        "warmup": warmup,
        "iterations": iterations,
        "prefill": prefill_summary,
        "decode_first": {
            "position": first_decode_position,
            "summary": first_decode_summary,
        },
        "decode_late": {
            "position": late_decode_position,
            "summary": late_decode_summary,
        },
        "estimated_generate_ms_using_first_decode": prefill_median_ms
        + decode_steps_for_max_new_tokens * first_decode_median_ms,
        "estimated_generate_ms_using_late_decode": prefill_median_ms
        + decode_steps_for_max_new_tokens * late_decode_median_ms,
        "decode_steps_for_max_new_tokens": decode_steps_for_max_new_tokens,
    }


def _decode_run_inputs(
    prefill_inputs: dict[str, np.ndarray],
    config,
    max_cache_len: int,
    *,
    next_id: int,
    position: int,
    cache: dict[str, np.ndarray],
    use_flash_static_kv_cache: bool,
    use_session_static_kv_cache: bool,
    use_decode_attention_mask: bool = False,
) -> dict[str, np.ndarray]:
    decode_inputs = {
        "input_ids": np.asarray([[next_id]], dtype=np.int64),
        "cos": _slice_position(prefill_inputs["text_cos"], position),
        "sin": _slice_position(prefill_inputs["text_sin"], position),
    }
    if use_decode_attention_mask or not use_flash_static_kv_cache:
        decode_inputs["attention_mask"] = _decode_attention_mask(
            config,
            max_cache_len,
            valid_past_len=position,
            use_flash_static_kv_cache=use_flash_static_kv_cache,
        )
    if not use_session_static_kv_cache:
        decode_inputs.update(cache)
    if use_flash_static_kv_cache and not use_session_static_kv_cache:
        decode_inputs["cache_seqlens"] = np.asarray([position], dtype=np.int32)
    return decode_inputs


def _run_prefill_for_decode(
    prefill_session,
    decode_session,
    prefill_run_inputs: dict[str, np.ndarray],
    spec: StaticKvCacheSpec,
    *,
    cache_len: int,
    use_session_static_kv_cache: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if not use_session_static_kv_cache:
        outputs = prefill_session.run_numpy(prefill_run_inputs)
        cache = seed_static_kv_cache(outputs, spec, cache_len=cache_len)
        return outputs, cache

    device_output_names = _present_output_names(spec)
    result = prefill_session.run_numpy_device_outputs(
        prefill_run_inputs,
        host_outputs=("logits",),
        device_outputs=device_output_names,
    )
    _copy_prefill_device_outputs_to_decode_state(decode_session, result, spec, cache_len=cache_len)
    decode_session.set_state_numpy("cache_seqlens", np.asarray([cache_len], dtype=np.int32))
    return {"logits": result["host_outputs"]["logits"]}, {}


def _present_output_names(spec: StaticKvCacheSpec) -> tuple[str, ...]:
    return tuple(
        name
        for layer_idx in range(spec.num_layers)
        for name in (spec.present_key_name(layer_idx), spec.present_value_name(layer_idx))
    )


def _copy_prefill_device_outputs_to_decode_state(
    decode_session,
    result: dict[str, object],
    spec: StaticKvCacheSpec,
    *,
    cache_len: int,
) -> None:
    device_outputs = result["device_outputs"]
    output_shapes = result["output_shapes"]
    if not isinstance(device_outputs, dict) or not isinstance(output_shapes, dict):
        raise TypeError("run_numpy_device_outputs result must contain device_outputs and output_shapes dictionaries")
    for layer_idx in range(spec.num_layers):
        _copy_present_output_to_state(
            decode_session,
            spec,
            device_outputs,
            output_shapes,
            present_name=spec.present_key_name(layer_idx),
            state_name=spec.past_key_name(layer_idx),
            cache_len=cache_len,
        )
        _copy_present_output_to_state(
            decode_session,
            spec,
            device_outputs,
            output_shapes,
            present_name=spec.present_value_name(layer_idx),
            state_name=spec.past_value_name(layer_idx),
            cache_len=cache_len,
        )


def _copy_present_output_to_state(
    decode_session,
    spec: StaticKvCacheSpec,
    device_outputs: dict[str, object],
    output_shapes: dict[str, object],
    *,
    present_name: str,
    state_name: str,
    cache_len: int,
) -> None:
    if present_name not in device_outputs:
        raise ValueError(f"Prefill did not return device output {present_name}")
    if present_name not in output_shapes:
        raise ValueError(f"Prefill did not report output shape for {present_name}")
    shape = tuple(int(dim) for dim in output_shapes[present_name])
    expected = tuple(spec.shape(cache_len=cache_len))
    if shape != expected:
        raise ValueError(f"Prefill output {present_name} has shape {shape}, expected {expected}")
    decode_session.copy_device_to_state_slice(
        state_name,
        int(device_outputs[present_name]),
        src_shape=shape,
        dst_start=(0, 0, 0, 0),
    )


def _cache_with_zero_filled_future(spec: StaticKvCacheSpec) -> dict[str, np.ndarray]:
    return empty_static_kv_cache(spec)


def _prefill_run_inputs(inputs: dict[str, np.ndarray], prompt_len: int) -> dict[str, np.ndarray]:
    run_inputs = {
        "input_ids": inputs["input_ids"],
        "pixel_values": inputs["pixel_values"],
    }
    if "attention_mask" in inputs:
        run_inputs["attention_mask"] = inputs["attention_mask"][:, :prompt_len, :prompt_len]
    return run_inputs


def _prefill_attention_mask(config, attention_mask: object | None, *, prompt_len: int) -> np.ndarray:
    mask_fill_value = float(getattr(config.text_config, "mask_fill_value", -1.0e4))
    mask = np.triu(
        np.full(
            (config.text_config.num_attention_heads, prompt_len, prompt_len),
            mask_fill_value,
            dtype=np.float32,
        ),
        k=1,
    )
    if attention_mask is not None:
        keep = np.asarray(attention_mask, dtype=bool)
        if keep.ndim != 2 or keep.shape[0] != 1 or keep.shape[1] < prompt_len:
            raise ValueError(f"attention_mask must have shape [1, >= {prompt_len}], got {keep.shape}")
        mask[:, :, ~keep[0, :prompt_len]] = mask_fill_value
    return _float_input(mask, config.text_config.dtype)


def _slice_position(values: np.ndarray, position: int) -> np.ndarray:
    return np.ascontiguousarray(values[:, int(position) : int(position) + 1, :])


def _decode_attention_mask(
    config,
    max_cache_len: int,
    *,
    valid_past_len: int,
    use_flash_static_kv_cache: bool,
) -> np.ndarray:
    if use_flash_static_kv_cache:
        mask = np.full((config.text_config.num_attention_heads, 1, max_cache_len), -1.0e4, dtype=np.float32)
        mask[:, :, : int(valid_past_len) + 1] = 0.0
        return _float_input(mask, config.text_config.dtype)
    mask = np.full((config.text_config.num_attention_heads, 1, max_cache_len + 1), -1.0e4, dtype=np.float32)
    mask[:, :, : int(valid_past_len)] = 0.0
    # The decode graph concatenates the new token after the fixed-size cache buffer.
    mask[:, :, max_cache_len] = 0.0
    return _float_input(mask, config.text_config.dtype)


def _float_input(values: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bfloat16":
        if values.dtype == np.uint16:
            return np.ascontiguousarray(values)
        return array_to_storage(values.astype(np.float32, copy=False), "bfloat16")
    return values.astype(dtype, copy=False)


def _use_flash_static_kv_cache(args: argparse.Namespace, config) -> bool:
    return (
        args.target in {"cuda", "rocm"}
        and config.text_config.dtype in {"float16", "bfloat16"}
        and bool(getattr(config.text_config, "use_flash_attention", True))
    )


def _use_session_static_kv_cache(args: argparse.Namespace, config) -> bool:
    return args.target == "rocm" and _use_flash_static_kv_cache(args, config)


def _enable_flash_attention_bias(config):
    text_config = config.text_config
    if bool(getattr(text_config, "use_flash_attention_bias", False)):
        return config
    if is_dataclass(config) and is_dataclass(text_config):
        return replace(config, text_config=replace(text_config, use_flash_attention_bias=True))
    setattr(text_config, "use_flash_attention_bias", True)
    return config


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    from transformers import AutoProcessor
    from transformers import __file__ as transformers_file
    from transformers import __version__ as transformers_version

    processor = AutoProcessor.from_pretrained(args.snapshot)
    processor_image_size = configure_processor_image_size(processor, args.min_pixels, args.max_pixels)
    processed, image_size, source_image_size = processor_inputs(
        processor,
        args.image,
        args.prompt,
        longest_side=args.longest_side,
    )
    config = build_config(args.snapshot, args.dtype)
    if args.attention_mask and args.target == "rocm":
        config = _enable_flash_attention_bias(config)
    inputs, prompt_len, max_cache_len, image_token_start = build_inputs(
        config,
        processed,
        args.max_new_tokens,
        use_attention_mask=args.attention_mask,
    )
    use_flash_static_kv_cache = _use_flash_static_kv_cache(args, config)
    use_session_static_kv_cache = _use_session_static_kv_cache(args, config)
    started_compile = time.perf_counter()
    prefill_artifact, decode_artifact = ensure_artifacts(args, config, inputs, image_token_start, max_cache_len)
    compile_seconds = time.perf_counter() - started_compile

    prefill_module = runtime.load(prefill_artifact, load_constants=True)
    decode_module = runtime.load(decode_artifact, load_constants=True)
    prefill_session = None
    decode_session = None
    try:
        prefill_session = prefill_module.create_session()
        decode_session = decode_module.create_session()
        eos_token_id = int(processor.tokenizer.eos_token_id)
        native_module_benchmark = benchmark_modules_native(
            prefill_session,
            decode_session,
            inputs,
            config,
            prompt_len,
            max_cache_len,
            args.max_new_tokens,
            warmup=args.benchmark_warmup,
            iterations=args.benchmark_iterations,
            use_flash_static_kv_cache=use_flash_static_kv_cache,
            use_session_static_kv_cache=use_session_static_kv_cache,
            use_decode_attention_mask=args.attention_mask,
        )
        times_ms: list[float] = []
        token_run_times_ms: list[list[float]] = []
        generated_ids: list[int] = []
        if not args.skip_generation:
            for _ in range(args.warmup):
                generate_once(
                    prefill_session,
                    decode_session,
                    inputs,
                    config,
                    prompt_len,
                    max_cache_len,
                    args.max_new_tokens,
                    eos_token_id,
                    use_flash_static_kv_cache=use_flash_static_kv_cache,
                    use_session_static_kv_cache=use_session_static_kv_cache,
                    use_decode_attention_mask=args.attention_mask,
                )
            for _ in range(args.iterations):
                started = time.perf_counter()
                generated_ids, per_token_times = generate_once(
                    prefill_session,
                    decode_session,
                    inputs,
                    config,
                    prompt_len,
                    max_cache_len,
                    args.max_new_tokens,
                    eos_token_id,
                    use_flash_static_kv_cache=use_flash_static_kv_cache,
                    use_session_static_kv_cache=use_session_static_kv_cache,
                    use_decode_attention_mask=args.attention_mask,
                )
                times_ms.append((time.perf_counter() - started) * 1000.0)
                token_run_times_ms.append(per_token_times)
    finally:
        if decode_session is not None:
            decode_session.close()
        if prefill_session is not None:
            prefill_session.close()
        decode_module.close()
        prefill_module.close()

    text = (
        ""
        if args.skip_generation
        else processor.post_process_image_text_to_text(np.asarray([generated_ids], dtype=np.int64), skip_special_tokens=True)[0]
    )
    payload = {
        "benchmark": "dinoml_glm_ocr_real_image_cached_generate",
        "prefill_artifact": str(prefill_artifact),
        "decode_artifact": str(decode_artifact),
        "snapshot": str(args.snapshot),
        "image": str(args.image),
        "source_image_size": list(source_image_size),
        "image_size": list(image_size),
        "longest_side": args.longest_side,
        "prompt": args.prompt,
        "transformers_version": transformers_version,
        "transformers_file": transformers_file,
        "dtype": args.dtype,
        "target": args.target,
        "arch": args.arch,
        "use_attention_mask": args.attention_mask,
        "prefill_uses_rope_constants": True,
        "use_flash_static_kv_cache": use_flash_static_kv_cache,
        "use_session_static_kv_cache": use_session_static_kv_cache,
        "processor_image_size": processor_image_size,
        "max_new_tokens": args.max_new_tokens,
        "prompt_len": prompt_len,
        "max_cache_len": max_cache_len,
        "generated_tokens": len(generated_ids),
        "generated_ids": generated_ids,
        "input_shapes": {name: list(value.shape) for name, value in inputs.items()},
        "image_token_start": image_token_start,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "skip_generation": args.skip_generation,
        "generation_wall_timing_note": (
            "times_ms and token_run_times_ms are Python parity-generation wall timings and include "
            "host-side cache/input/output handling; use native_module_benchmark for performance."
        ),
        "native_module_benchmark": native_module_benchmark,
        "compile_seconds": compile_seconds,
        "times_ms": times_ms,
        "token_run_times_ms": token_run_times_ms,
        "median_ms": None if not times_ms else statistics.median(times_ms),
        "mean_ms": None if not times_ms else statistics.fmean(times_ms),
        "min_ms": None if not times_ms else min(times_ms),
        "max_ms": None if not times_ms else max(times_ms),
        "text": text,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
