from __future__ import annotations

import argparse
import gc
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from dinoml import runtime
from dinoml.models.kv_cache import StaticKvCacheSpec
from dinoml.models.qwen2_5_vl import (
    qwen2_5_vl_rope_index,
    qwen2_5_vl_text_rope_embeddings,
    qwen2_5_vl_vision_cu_seqlens,
    qwen2_5_vl_vision_position_ids,
    qwen2_5_vl_vision_rope_embeddings,
    qwen2_5_vl_vision_window_index,
)
from dinoml.models.qwen2_5_vl.workflow_common import (
    float_input,
    load_qwen2_5_vl_config,
    load_qwen2_5_vl_weights,
)

try:
    from tools.qwen2_5_vl_benchmark_common import (
        DEFAULT_IMAGE,
        DEFAULT_PROMPT,
        DEFAULT_SNAPSHOT,
        configure_processor_image_size,
        open_rgb_image,
    )
except ModuleNotFoundError:
    from qwen2_5_vl_benchmark_common import (
        DEFAULT_IMAGE,
        DEFAULT_PROMPT,
        DEFAULT_SNAPSHOT,
        configure_processor_image_size,
        open_rgb_image,
    )


TRANSFORMERS_REFERENCE_INPUTS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
        "second_per_grid_ts",
        "mm_token_type_ids",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and optionally benchmark a Qwen2.5-VL image-prefill + static-cache decode pipeline."
    )
    parser.add_argument("--compare-json", nargs=2, metavar=("DINOML_JSON", "TRANSFORMERS_JSON"), default=None)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prefill-artifact", type=Path, default=None)
    parser.add_argument("--decode-artifact", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--longest-side", type=int, default=None)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--compare-transformers", action="store_true")
    parser.add_argument("--transformers-device", default="cuda")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def load_stop_token_ids(snapshot: Path, processor) -> tuple[int, ...]:
    from transformers import GenerationConfig

    stop_ids: list[int] = []
    try:
        generation_config = GenerationConfig.from_pretrained(snapshot)
        raw_eos = generation_config.eos_token_id
        if isinstance(raw_eos, int):
            stop_ids.append(int(raw_eos))
        elif raw_eos is not None:
            stop_ids.extend(int(value) for value in raw_eos)
    except OSError:
        pass
    tokenizer_eos = getattr(processor.tokenizer, "eos_token_id", None)
    if tokenizer_eos is not None:
        stop_ids.append(int(tokenizer_eos))
    return tuple(dict.fromkeys(stop_ids))


def load_processor_and_inputs(
    snapshot: Path,
    image_path: Path,
    prompt: str,
    *,
    longest_side: int | None,
    min_pixels: int | None,
    max_pixels: int | None,
):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(snapshot)
    processor_image_size = configure_processor_image_size(processor, min_pixels, max_pixels)
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
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    try:
        processed_torch = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            return_mm_token_type_ids=True,
        )
    except TypeError:
        processed_torch = processor(text=[text], images=[image], return_tensors="pt")
    processed_numpy = {name: value.detach().cpu().numpy() for name, value in processed_torch.items()}
    return processor, processed_torch, processed_numpy, image.size, source_image_size, processor_image_size


def build_pipeline_inputs(
    config,
    processed: dict[str, np.ndarray],
    max_new_tokens: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int, int, np.ndarray]:
    input_ids = np.asarray(processed["input_ids"], dtype=np.int64)
    image_grid_thw = np.asarray(processed["image_grid_thw"], dtype=np.int64)
    pixel_values = np.asarray(processed["pixel_values"], dtype=np.float32)
    mm_token_type_ids = processed.get("mm_token_type_ids")
    if mm_token_type_ids is None:
        mm_token_type_ids = (input_ids == int(config.image_token_id)).astype(np.int64)
    else:
        mm_token_type_ids = np.asarray(mm_token_type_ids, dtype=np.int64)
    text_attention_mask = processed.get("attention_mask")
    if text_attention_mask is not None:
        text_attention_mask = np.asarray(text_attention_mask, dtype=np.int64)
    prefill_len = int(input_ids.shape[1])
    max_cache_len = prefill_len + int(max_new_tokens)

    prefill_position_ids, rope_deltas = qwen2_5_vl_rope_index(
        input_ids,
        mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=text_attention_mask,
        spatial_merge_size=config.vision_config.spatial_merge_size,
    )
    decode_text_positions = np.arange(prefill_len, max_cache_len, dtype=np.int64).reshape(1, 1, max_new_tokens)
    decode_text_positions = np.broadcast_to(decode_text_positions, (3, input_ids.shape[0], max_new_tokens)).copy()
    decode_text_positions += np.asarray(rope_deltas, dtype=np.int64).reshape(1, input_ids.shape[0], 1)
    text_position_ids = np.concatenate([prefill_position_ids, decode_text_positions], axis=2)
    text_cos, text_sin = qwen2_5_vl_text_rope_embeddings(text_position_ids, config.text_config, dtype=config.text_config.dtype)
    vision_position_ids = qwen2_5_vl_vision_position_ids(image_grid_thw, config.vision_config.spatial_merge_size)
    vision_cos, vision_sin = qwen2_5_vl_vision_rope_embeddings(vision_position_ids, head_dim=config.vision_config.head_dim)
    window_index, window_cu = qwen2_5_vl_vision_window_index(
        image_grid_thw,
        spatial_merge_size=config.vision_config.spatial_merge_size,
        window_size=config.vision_config.window_size,
        patch_size=config.vision_config.patch_size,
    )
    reverse_window_index = np.argsort(window_index).astype(np.int32)
    full_cu = qwen2_5_vl_vision_cu_seqlens(image_grid_thw).astype(np.int32)
    pixel_values = window_order_merged(pixel_values, window_index, config.vision_config.spatial_merge_unit)
    vision_cos = window_order_merged(vision_cos, window_index, config.vision_config.spatial_merge_unit)
    vision_sin = window_order_merged(vision_sin, window_index, config.vision_config.spatial_merge_unit)

    full_inputs = {
        "input_ids": input_ids.astype(np.int32, copy=False),
        "pixel_values": float_input(pixel_values, config.vision_config.dtype),
        "image_grid_thw": image_grid_thw.astype(np.float32, copy=False),
        "vision_cos": float_input(vision_cos, "float32"),
        "vision_sin": float_input(vision_sin, "float32"),
        "vision_full_cu_seqlens": full_cu,
        "vision_window_cu_seqlens": window_cu.astype(np.int32),
        "vision_reverse_window_index": reverse_window_index,
        "text_cos": float_input(text_cos, config.text_config.dtype),
        "text_sin": float_input(text_sin, config.text_config.dtype),
        "attention_mask": prefill_attention_mask(config, processed.get("attention_mask"), prompt_len=prefill_len),
    }
    prefill_inputs = {
        "input_ids": full_inputs["input_ids"],
        "pixel_values": full_inputs["pixel_values"],
        "image_grid_thw": full_inputs["image_grid_thw"],
        "vision_cos": full_inputs["vision_cos"],
        "vision_sin": full_inputs["vision_sin"],
        "vision_full_cu_seqlens": full_inputs["vision_full_cu_seqlens"],
        "vision_window_cu_seqlens": full_inputs["vision_window_cu_seqlens"],
        "vision_reverse_window_index": full_inputs["vision_reverse_window_index"],
        "text_cos": np.ascontiguousarray(full_inputs["text_cos"][:, :prefill_len, :]),
        "text_sin": np.ascontiguousarray(full_inputs["text_sin"][:, :prefill_len, :]),
        "attention_mask": full_inputs["attention_mask"],
    }
    return prefill_inputs, full_inputs, prefill_len, max_cache_len, rope_deltas


def window_order_merged(values: np.ndarray, window_index: np.ndarray, merge_unit: int) -> np.ndarray:
    value = np.asarray(values)
    grouped = value.reshape(-1, int(merge_unit), value.shape[-1])
    return np.ascontiguousarray(grouped[np.asarray(window_index, dtype=np.int64)].reshape(value.shape))


def cache_spec_for_config(config, max_cache_len: int) -> StaticKvCacheSpec:
    return StaticKvCacheSpec(
        num_layers=config.text_config.num_hidden_layers,
        batch=1,
        num_key_value_heads=config.text_config.num_key_value_heads,
        max_cache_len=max_cache_len,
        head_dim=config.text_config.head_dim,
        dtype=config.text_config.dtype,
    )


def load_artifact_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Artifact metadata not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_artifact_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Artifact manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def inspect_compiled_artifact(path: Path) -> dict[str, Any]:
    manifest = load_artifact_manifest(path)
    metadata = load_artifact_metadata(path)
    files = manifest.get("files", {})
    if not isinstance(files, dict):
        raise ValueError(f"Artifact manifest {path / 'manifest.json'} must expose a files mapping")
    missing_files = sorted(
        file_name
        for file_name in files.values()
        if isinstance(file_name, str) and not (path / file_name).is_file()
    )
    if missing_files:
        raise FileNotFoundError(
            f"Artifact {path} is missing manifest-declared files: {', '.join(missing_files)}"
        )
    target_name = str((manifest.get("target") or {}).get("name", ""))
    if target_name not in {"rocm", "cuda"}:
        raise ValueError(f"Artifact {path} target must be a GPU backend, got {target_name!r}")
    return {
        "path": str(path),
        "status": "compiled",
        "target": target_name,
        "runtime_abi_version": manifest.get("runtime_abi_version"),
        "module": str(path / str(files.get("module", "module.so"))),
        "metadata_inputs": [entry["name"] for entry in metadata.get("inputs", [])],
        "metadata_outputs": [entry["name"] for entry in metadata.get("outputs", [])],
    }


def validate_artifacts(*, prefill_artifact: Path, decode_artifact: Path) -> dict[str, Any]:
    prefill_info = inspect_compiled_artifact(prefill_artifact)
    decode_info = inspect_compiled_artifact(decode_artifact)
    if prefill_info["target"] != decode_info["target"]:
        raise ValueError(
            f"Prefill and decode artifacts must target the same backend, got {prefill_info['target']!r} and {decode_info['target']!r}"
        )
    prefill_metadata = load_artifact_metadata(prefill_artifact)
    decode_metadata = load_artifact_metadata(decode_artifact)
    prefill_inputs = {entry["name"] for entry in prefill_metadata.get("inputs", [])}
    prefill_outputs = {entry["name"] for entry in prefill_metadata.get("outputs", [])}
    decode_inputs = {entry["name"] for entry in decode_metadata.get("inputs", [])}
    decode_outputs = {entry["name"] for entry in decode_metadata.get("outputs", [])}

    required_prefill_inputs = {
        "input_ids",
        "pixel_values",
        "image_grid_thw",
        "vision_cos",
        "vision_sin",
        "vision_full_cu_seqlens",
        "vision_window_cu_seqlens",
        "vision_reverse_window_index",
        "text_cos",
        "text_sin",
        "attention_mask",
    }
    missing_prefill = sorted(required_prefill_inputs - prefill_inputs)
    if missing_prefill:
        raise ValueError(f"Prefill artifact {prefill_artifact} is missing expected inputs: {', '.join(missing_prefill)}")
    if "logits" not in prefill_outputs:
        raise ValueError(f"Prefill artifact {prefill_artifact} does not expose logits")
    if not any(name.startswith("present_key_") for name in prefill_outputs):
        raise ValueError(f"Prefill artifact {prefill_artifact} does not expose cache outputs")

    required_decode_inputs = {"input_ids", "cos", "sin"}
    missing_decode = sorted(required_decode_inputs - decode_inputs)
    if missing_decode:
        raise ValueError(f"Decode artifact {decode_artifact} is missing expected inputs: {', '.join(missing_decode)}")
    if "logits" not in decode_outputs:
        raise ValueError(f"Decode artifact {decode_artifact} does not expose logits")
    if not any(name.startswith("past_key_") for name in decode_inputs):
        raise ValueError(f"Decode artifact {decode_artifact} does not expose decode cache inputs")
    if any(name.startswith("new_key_") for name in decode_outputs):
        if "cache_seqlens" not in decode_inputs:
            raise ValueError(f"Decode artifact {decode_artifact} exposes static KV cache updates but is missing cache_seqlens")
        return {
            "status": "validated",
            "requested": True,
            "compiled": True,
            "executed": False,
            "prefill": prefill_info,
            "decode": decode_info,
            "decode_mode": "static",
            "use_decode_attention_mask": "attention_mask" in decode_inputs,
        }
    if any(name.startswith("present_key_") for name in decode_outputs):
        return {
            "status": "validated",
            "requested": True,
            "compiled": True,
            "executed": False,
            "prefill": prefill_info,
            "decode": decode_info,
            "decode_mode": "dynamic",
            "use_decode_attention_mask": "attention_mask" in decode_inputs,
        }
    raise ValueError(f"Decode artifact {decode_artifact} must expose either present_key_* outputs or new_key_* outputs")


def prefill_attention_mask(config, attention_mask: object | None, *, prompt_len: int) -> np.ndarray:
    mask_fill_value = float(getattr(config.text_config, "mask_fill_value", -1.0e4))
    mask = np.triu(
        np.full((config.text_config.num_attention_heads, prompt_len, prompt_len), mask_fill_value, dtype=np.float32),
        k=1,
    )
    if attention_mask is not None:
        keep = np.asarray(attention_mask, dtype=bool)
        if keep.ndim != 2 or keep.shape[0] != 1 or keep.shape[1] < prompt_len:
            raise ValueError(f"attention_mask must have shape [1, >= {prompt_len}], got {keep.shape}")
        mask[:, :, ~keep[0, :prompt_len]] = mask_fill_value
    return float_input(mask, config.text_config.dtype)


def can_use_device_pointers(module) -> bool:
    return module.target_name in {"cuda", "rocm"}


def torch_tensor_from_storage(value: np.ndarray, device, *, torch):
    array = np.asarray(value)
    tensor = torch.from_numpy(array)
    if array.dtype == np.uint16:
        tensor = tensor.view(torch.bfloat16)
    return tensor.to(device=device)


def torch_empty(shape: tuple[int, ...], dtype_name: str, device, *, torch):
    return torch.empty(shape, dtype=torch_dtype(dtype_name), device=device)


def torch_inputs_from_numpy(inputs: dict[str, np.ndarray], device, *, torch) -> dict[str, object]:
    return {
        name: torch_tensor_from_storage(value, device, torch=torch)
        for name, value in inputs.items()
    }


def prefill_output_shapes(config, prefill_len: int) -> dict[str, tuple[int, ...]]:
    shapes = {
        "logits": (1, 1, int(config.text_config.vocab_size)),
    }
    for layer_idx in range(int(config.text_config.num_hidden_layers)):
        cache_shape = (
            1,
            int(config.text_config.num_key_value_heads),
            int(prefill_len),
            int(config.text_config.head_dim),
        )
        shapes[f"present_key_{layer_idx}"] = cache_shape
        shapes[f"present_value_{layer_idx}"] = cache_shape
    return shapes


def decode_output_buffer_shapes(config, *, decode_mode: str, max_cache_len: int) -> dict[str, tuple[int, ...]]:
    shapes = {
        "logits": (1, 1, int(config.text_config.vocab_size)),
    }
    if decode_mode == "static":
        update_shape = (
            1,
            int(config.text_config.num_key_value_heads),
            1,
            int(config.text_config.head_dim),
        )
    else:
        update_shape = (
            1,
            int(config.text_config.num_key_value_heads),
            int(max_cache_len),
            int(config.text_config.head_dim),
        )
    for layer_idx in range(int(config.text_config.num_hidden_layers)):
        prefix = "new" if decode_mode == "static" else "present"
        shapes[f"{prefix}_key_{layer_idx}"] = update_shape
        shapes[f"{prefix}_value_{layer_idx}"] = update_shape
    return shapes


def decode_output_shapes(config, *, decode_mode: str, cache_len: int) -> dict[str, tuple[int, ...]]:
    shapes = {
        "logits": (1, 1, int(config.text_config.vocab_size)),
    }
    if decode_mode == "static":
        update_shape = (
            1,
            int(config.text_config.num_key_value_heads),
            1,
            int(config.text_config.head_dim),
        )
        for layer_idx in range(int(config.text_config.num_hidden_layers)):
            shapes[f"new_key_{layer_idx}"] = update_shape
            shapes[f"new_value_{layer_idx}"] = update_shape
        return shapes
    present_shape = (
        1,
        int(config.text_config.num_key_value_heads),
        int(cache_len),
        int(config.text_config.head_dim),
    )
    for layer_idx in range(int(config.text_config.num_hidden_layers)):
        shapes[f"present_key_{layer_idx}"] = present_shape
        shapes[f"present_value_{layer_idx}"] = present_shape
    return shapes


def prepare_device_pipeline(
    *,
    prefill_inputs: dict[str, np.ndarray],
    full_inputs: dict[str, np.ndarray],
    config,
    decode_mode: str,
    prefill_len: int,
    max_cache_len: int,
    device,
):
    import torch

    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        prefill_inputs_t = torch_inputs_from_numpy(prefill_inputs, device, torch=torch)
        full_inputs_t = torch_inputs_from_numpy(full_inputs, device, torch=torch)
        prefill_shapes = prefill_output_shapes(config, prefill_len)
        prefill_outputs_t = {
            name: torch_empty(shape, config.text_config.dtype, device, torch=torch)
            for name, shape in prefill_shapes.items()
        }
        if decode_mode == "static":
            cache_shape = tuple(cache_spec_for_config(config, max_cache_len).shape())
            cache_t = {
                name: torch.zeros(
                    cache_shape,
                    dtype=torch_dtype(config.text_config.dtype),
                    device=device,
                )
                for layer_idx in range(int(config.text_config.num_hidden_layers))
                for name in (f"past_key_{layer_idx}", f"past_value_{layer_idx}")
            }
        else:
            cache_t = {}
        decode_output_shapes_map = decode_output_buffer_shapes(
            config, decode_mode=decode_mode, max_cache_len=max_cache_len
        )
        if decode_mode == "static":
            decode_outputs_t = {
                name: torch_empty(shape, config.text_config.dtype, device, torch=torch)
                for name, shape in decode_output_shapes_map.items()
            }
        else:
            decode_outputs_t = {}
        attention_mask_t = None
        if "attention_mask" in prefill_inputs:
            attention_mask_t = torch.zeros(
                (int(config.text_config.num_attention_heads), 1, int(max_cache_len)),
                dtype=torch_dtype(config.text_config.dtype),
                device=device,
            )
    stream.synchronize()
    return {
        "device": device,
        "stream": stream,
        "prefill_inputs_t": prefill_inputs_t,
        "full_inputs_t": full_inputs_t,
        "prefill_shapes": prefill_shapes,
        "prefill_outputs_t": prefill_outputs_t,
        "cache_t": cache_t,
        "decode_output_shapes_map": decode_output_shapes_map,
        "decode_mode": decode_mode,
        "decode_outputs_t": decode_outputs_t,
        "input_ids_t": torch.empty((1, 1), dtype=torch.int32, device=device),
        "cos_t": torch.empty((1, 1, int(config.text_config.head_dim)), dtype=torch_dtype(config.text_config.dtype), device=device),
        "sin_t": torch.empty((1, 1, int(config.text_config.head_dim)), dtype=torch_dtype(config.text_config.dtype), device=device),
        "cache_seqlens_t": torch.empty((1,), dtype=torch.int32, device=device),
        "attention_mask_t": attention_mask_t,
    }


def release_device_runtime_memory(*, torch) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def run_prefill_device(
    prefill_session,
    *,
    config,
    device_ctx: dict[str, Any],
    prefill_len: int,
) -> tuple[int, dict[str, Any]]:
    import torch

    stream = device_ctx["stream"]
    prefill_inputs_t = device_ctx["prefill_inputs_t"]
    prefill_shapes = device_ctx["prefill_shapes"]
    prefill_outputs_t = device_ctx["prefill_outputs_t"]
    decode_mode = str(device_ctx["decode_mode"])
    prefill_session.set_stream(stream)
    with torch.cuda.stream(stream):
        prefill_session.run_device_pointers(
            {name: int(tensor.data_ptr()) for name, tensor in prefill_inputs_t.items()},
            {name: int(tensor.data_ptr()) for name, tensor in prefill_outputs_t.items()},
            {name: tuple(int(dim) for dim in tensor.shape) for name, tensor in prefill_inputs_t.items()},
            prefill_shapes,
        )

        cache_t = device_ctx["cache_t"]
        if decode_mode == "static":
            for layer_idx in range(int(config.text_config.num_hidden_layers)):
                cache_t[f"past_key_{layer_idx}"][:, :, :prefill_len, :].copy_(
                    prefill_outputs_t[f"present_key_{layer_idx}"]
                )
                cache_t[f"past_value_{layer_idx}"][:, :, :prefill_len, :].copy_(
                    prefill_outputs_t[f"present_value_{layer_idx}"]
                )
        else:
            cache_t = {
                f"past_key_{layer_idx}": prefill_outputs_t[f"present_key_{layer_idx}"].clone()
                for layer_idx in range(int(config.text_config.num_hidden_layers))
            } | {
                f"past_value_{layer_idx}": prefill_outputs_t[f"present_value_{layer_idx}"].clone()
                for layer_idx in range(int(config.text_config.num_hidden_layers))
            }
        device_ctx["cache_t"] = cache_t

        next_id = int(torch.argmax(prefill_outputs_t["logits"][0, 0, :]).item())
    stream.synchronize()
    logits = prefill_outputs_t["logits"].detach().float().cpu().numpy()
    return next_id, {
        "logits_shape": list(prefill_outputs_t["logits"].shape),
        "argmax_token_id": next_id,
        "logits": logits,
    }


def run_decode_device(
    decode_session,
    *,
    config,
    device_ctx: dict[str, Any],
    initial_next_id: int,
    use_decode_attention_mask: bool,
    prefill_len: int,
    max_cache_len: int,
    max_new_tokens: int,
    stop_token_ids: tuple[int, ...],
) -> tuple[list[int], list[float], dict[str, Any]]:
    import torch

    device = device_ctx["device"]
    stream = device_ctx["stream"]
    full_inputs_t = device_ctx["full_inputs_t"]
    decode_outputs_t = device_ctx["decode_outputs_t"]
    input_ids_t = device_ctx["input_ids_t"]
    cos_t = device_ctx["cos_t"]
    sin_t = device_ctx["sin_t"]
    cache_seqlens_t = device_ctx["cache_seqlens_t"]
    attention_mask_t = device_ctx["attention_mask_t"]
    decode_mode = str(device_ctx["decode_mode"])
    mask_fill_value = float(getattr(config.text_config, "mask_fill_value", -1.0e4))
    decode_session.set_stream(stream)
    cache_t = device_ctx["cache_t"]
    decode_times_ms: list[float] = []
    next_id = int(initial_next_id)
    first_decode_probe: dict[str, Any] = {"status": "not_run", "reason": "generation_stopped_after_prefill"}
    with torch.cuda.stream(stream):
        generated_ids: list[int] = []
        for step in range(max_new_tokens):
            generated_ids.append(next_id)
            if next_id in stop_token_ids or step == max_new_tokens - 1:
                break
            position = prefill_len + step
            total_len = position + 1
            input_ids_t.fill_(next_id)
            cos_t.copy_(full_inputs_t["text_cos"][:, position : position + 1, :])
            sin_t.copy_(full_inputs_t["text_sin"][:, position : position + 1, :])
            decode_inputs_t: dict[str, object] = {
                "input_ids": input_ids_t,
                "cos": cos_t,
                "sin": sin_t,
                **cache_t,
            }
            decode_input_shapes = {
                "input_ids": (1, 1),
                "cos": (1, 1, int(config.text_config.head_dim)),
                "sin": (1, 1, int(config.text_config.head_dim)),
            }
            static_cache_len = int(max_cache_len)
            if decode_mode == "static":
                cache_seqlens_t.fill_(position)
                decode_inputs_t["cache_seqlens"] = cache_seqlens_t
                decode_input_shapes["cache_seqlens"] = (1,)
            for layer_idx in range(int(config.text_config.num_hidden_layers)):
                cache_len = static_cache_len if decode_mode == "static" else position
                decode_input_shapes[f"past_key_{layer_idx}"] = (
                    1,
                    int(config.text_config.num_key_value_heads),
                    cache_len,
                    int(config.text_config.head_dim),
                )
                decode_input_shapes[f"past_value_{layer_idx}"] = (
                    1,
                    int(config.text_config.num_key_value_heads),
                    cache_len,
                    int(config.text_config.head_dim),
                )
            if use_decode_attention_mask and attention_mask_t is not None:
                attention_mask_t.fill_(mask_fill_value)
                attention_mask_t[:, :, :total_len].fill_(0.0)
                decode_inputs_t["attention_mask"] = attention_mask_t
                mask_len = static_cache_len if decode_mode == "static" else total_len
                decode_input_shapes["attention_mask"] = (
                    int(config.text_config.num_attention_heads),
                    1,
                    mask_len,
                )
            stream.synchronize()
            started = time.perf_counter()
            if decode_mode == "static":
                step_decode_outputs_t = decode_outputs_t
            else:
                step_decode_outputs_t = {
                    name: torch_empty(
                        shape,
                        config.text_config.dtype,
                        device,
                        torch=torch,
                    )
                    for name, shape in decode_output_shapes(config, decode_mode=decode_mode, cache_len=total_len).items()
                }
            decode_session.run_device_pointers(
                {name: int(tensor.data_ptr()) for name, tensor in decode_inputs_t.items()},
                {name: int(tensor.data_ptr()) for name, tensor in step_decode_outputs_t.items()},
                decode_input_shapes,
                decode_output_shapes(config, decode_mode=decode_mode, cache_len=total_len),
            )
            stream.synchronize()
            decode_times_ms.append((time.perf_counter() - started) * 1000.0)
            if step == 0:
                first_decode_logits = step_decode_outputs_t["logits"].detach().float().cpu().numpy()
                first_decode_probe = {
                    "status": "ok",
                    "input_token_id": int(next_id),
                    "argmax_token_id": int(torch.argmax(step_decode_outputs_t["logits"][0, 0, :]).item()),
                    "logits_shape": list(step_decode_outputs_t["logits"].shape),
                    "logits": first_decode_logits,
                }
            if decode_mode == "static":
                for layer_idx in range(int(config.text_config.num_hidden_layers)):
                    cache_t[f"past_key_{layer_idx}"][:, :, position : position + 1, :].copy_(
                        step_decode_outputs_t[f"new_key_{layer_idx}"]
                    )
                    cache_t[f"past_value_{layer_idx}"][:, :, position : position + 1, :].copy_(
                        step_decode_outputs_t[f"new_value_{layer_idx}"]
                    )
            else:
                cache_t = {
                    f"past_key_{layer_idx}": step_decode_outputs_t[f"present_key_{layer_idx}"]
                    for layer_idx in range(int(config.text_config.num_hidden_layers))
                } | {
                    f"past_value_{layer_idx}": step_decode_outputs_t[f"present_value_{layer_idx}"]
                    for layer_idx in range(int(config.text_config.num_hidden_layers))
                }
            next_id = int(torch.argmax(step_decode_outputs_t["logits"][0, 0, :]).item())

    stream.synchronize()
    return generated_ids, decode_times_ms, first_decode_probe


def run_artifacts(
    *,
    prefill_artifact: Path,
    decode_artifact: Path,
    artifact_modes: dict[str, Any],
    processor,
    prefill_inputs: dict[str, np.ndarray],
    full_inputs: dict[str, np.ndarray],
    config,
    prefill_len: int,
    max_cache_len: int,
    max_new_tokens: int,
    stop_token_ids: tuple[int, ...],
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Qwen2.5-VL artifact benchmark requires torch for GPU device-pointer execution") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen2.5-VL artifact benchmark requires a ROCm/CUDA device exposed through torch.cuda")

    device = torch.device("cuda")
    device_ctx = prepare_device_pipeline(
        prefill_inputs=prefill_inputs,
        full_inputs=full_inputs,
        config=config,
        decode_mode=str(artifact_modes["decode_mode"]),
        prefill_len=prefill_len,
        max_cache_len=max_cache_len,
        device=device,
    )
    execution_mode = "device_pointers_sequential_modules"
    generated_ids: list[int] = []
    pipeline_times_ms: list[float] = []
    decode_step_runs_ms: list[list[float]] = []
    prefill_output_preview: dict[str, Any] = {}
    first_decode_output_preview: dict[str, Any] = {"status": "not_run"}

    def run_once(*, record_timing: bool) -> None:
        nonlocal generated_ids, prefill_output_preview, first_decode_output_preview
        prefill_module = runtime.load(prefill_artifact, load_constants=True)
        prefill_session = None
        try:
            if not can_use_device_pointers(prefill_module):
                raise RuntimeError(f"Prefill artifact {prefill_artifact} is not a GPU device-pointer artifact")
            prefill_session = prefill_module.create_session()
            started = time.perf_counter()
            next_id, prefill_outputs = run_prefill_device(
                prefill_session,
                config=config,
                device_ctx=device_ctx,
                prefill_len=prefill_len,
            )
            prefill_time_ms = (time.perf_counter() - started) * 1000.0
            prefill_output_preview = {
                "logits_shape": list(prefill_outputs["logits_shape"]),
                "argmax_token_id": int(prefill_outputs["argmax_token_id"]),
            }
        finally:
            if prefill_session is not None:
                prefill_session.close()
            prefill_module.close()
            release_device_runtime_memory(torch=torch)

        decode_module = runtime.load(decode_artifact, load_constants=True)
        decode_session = None
        try:
            if not can_use_device_pointers(decode_module):
                raise RuntimeError(f"Decode artifact {decode_artifact} is not a GPU device-pointer artifact")
            decode_session = decode_module.create_session()
            started = time.perf_counter()
            generated_ids, decode_times, decode_probe = run_decode_device(
                decode_session,
                config=config,
                device_ctx=device_ctx,
                initial_next_id=next_id,
                use_decode_attention_mask=bool(artifact_modes["use_decode_attention_mask"]),
                prefill_len=prefill_len,
                max_cache_len=max_cache_len,
                max_new_tokens=max_new_tokens,
                stop_token_ids=stop_token_ids,
            )
            decode_time_ms = (time.perf_counter() - started) * 1000.0
        finally:
            if decode_session is not None:
                decode_session.close()
            decode_module.close()
            release_device_runtime_memory(torch=torch)

        if record_timing:
            pipeline_times_ms.append(prefill_time_ms + decode_time_ms)
            decode_step_runs_ms.append(decode_times)
        if decode_probe.get("status") == "ok":
            first_decode_output_preview = {
                "status": "ok",
                "input_token_id": int(decode_probe["input_token_id"]),
                "argmax_token_id": int(decode_probe["argmax_token_id"]),
                "logits_shape": list(decode_probe["logits_shape"]),
            }
        else:
            first_decode_output_preview = dict(decode_probe)
        device_ctx["_prefill_logits"] = prefill_outputs["logits"]
        device_ctx["_first_decode_logits"] = decode_probe.get("logits")

    for _ in range(int(warmup)):
        run_once(record_timing=False)
    for _ in range(int(iterations)):
        run_once(record_timing=True)

    decoded = processor.batch_decode(
        np.asarray([generated_ids], dtype=np.int64),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return {
        "status": "ok",
        "prefill_artifact": str(prefill_artifact),
        "decode_artifact": str(decode_artifact),
        "decode_mode": str(artifact_modes["decode_mode"]),
        "use_decode_attention_mask": bool(artifact_modes["use_decode_attention_mask"]),
        "warmup": int(warmup),
        "iterations": int(iterations),
        "execution_mode": execution_mode,
        "module_residency": "prefill_unloaded_before_decode",
        "timing_excludes_module_load_unload": True,
        "generated_token_count": len(generated_ids),
        "generated_ids": generated_ids,
        "text": decoded[0] if decoded else "",
        "pipeline_times_ms": pipeline_times_ms,
        "decode_step_runs_ms": decode_step_runs_ms,
        "prefill_output_preview": prefill_output_preview,
        "first_decode_output_preview": first_decode_output_preview,
        "_prefill_logits": device_ctx.get("_prefill_logits"),
        "_first_decode_logits": device_ctx.get("_first_decode_logits"),
    }


def compare_generated_outputs(dinoml: dict[str, Any], transformers: dict[str, Any]) -> dict[str, Any]:
    if dinoml.get("status") != "ok" or transformers.get("status") != "ok":
        return {
            "status": "not_compared",
            "reason": "both_dinoml_and_transformers_must_be_ok",
        }
    dinoml_ids = [int(value) for value in dinoml.get("generated_ids", [])]
    transformers_ids = [int(value) for value in transformers.get("generated_ids", [])]
    first_mismatch_index: int | None = None
    for index, (dinoml_id, transformers_id) in enumerate(zip(dinoml_ids, transformers_ids)):
        if dinoml_id != transformers_id:
            first_mismatch_index = index
            break
    if first_mismatch_index is None and len(dinoml_ids) != len(transformers_ids):
        first_mismatch_index = min(len(dinoml_ids), len(transformers_ids))
    generated_ids_match = first_mismatch_index is None
    text_matches = str(dinoml.get("text", "")) == str(transformers.get("text", ""))
    result: dict[str, Any] = {
        "status": "ok" if generated_ids_match and text_matches else "mismatch",
        "generated_ids_match": generated_ids_match,
        "text_matches": text_matches,
        "dinoml_generated_token_count": len(dinoml_ids),
        "transformers_generated_token_count": len(transformers_ids),
    }
    if first_mismatch_index is not None:
        result.update(
            {
                "first_mismatch_index": int(first_mismatch_index),
                "dinoml_token": dinoml_ids[first_mismatch_index] if first_mismatch_index < len(dinoml_ids) else None,
                "transformers_token": transformers_ids[first_mismatch_index]
                if first_mismatch_index < len(transformers_ids)
                else None,
            }
        )
    return result


def compare_logits_outputs(
    dinoml: dict[str, Any],
    transformers: dict[str, Any],
    *,
    dinoml_key: str,
    transformers_key: str,
    compare_name: str,
    rtol: float = 3.0e-2,
    atol: float = 3.0e-2,
) -> dict[str, Any]:
    if dinoml.get("status") != "ok" or transformers.get("status") != "ok":
        return {
            "status": "not_compared",
            "reason": "both_dinoml_and_transformers_must_be_ok",
            "surface": compare_name,
        }
    dinoml_logits = dinoml.get(dinoml_key)
    transformers_logits = transformers.get(transformers_key)
    if dinoml_logits is None or transformers_logits is None:
        return {
            "status": "not_compared",
            "reason": "missing_probe_logits",
            "surface": compare_name,
        }
    lhs = np.asarray(dinoml_logits, dtype=np.float32)
    rhs = np.asarray(transformers_logits, dtype=np.float32)
    if lhs.shape != rhs.shape:
        return {
            "status": "mismatch",
            "surface": compare_name,
            "reason": "shape_mismatch",
            "dinoml_shape": list(lhs.shape),
            "transformers_shape": list(rhs.shape),
        }
    diff = np.abs(lhs - rhs)
    max_abs_diff = float(diff.max(initial=0.0))
    allclose = bool(np.allclose(lhs, rhs, rtol=rtol, atol=atol))
    dinoml_argmax = int(np.argmax(lhs.reshape(-1)))
    transformers_argmax = int(np.argmax(rhs.reshape(-1)))
    argmax_match = dinoml_argmax == transformers_argmax
    return {
        "status": "ok" if allclose and argmax_match else "mismatch",
        "surface": compare_name,
        "shape": list(lhs.shape),
        "allclose": allclose,
        "argmax_match": argmax_match,
        "dinoml_argmax": dinoml_argmax,
        "transformers_argmax": transformers_argmax,
        "max_abs_diff": max_abs_diff,
        "rtol": rtol,
        "atol": atol,
    }


def build_verification_summary(
    *,
    artifacts: dict[str, Any],
    dinoml: dict[str, Any],
    transformers: dict[str, Any],
) -> dict[str, Any]:
    if artifacts.get("status") == "not_requested":
        artifact_compilation = {"status": "not_requested"}
    elif artifacts.get("status") == "invalid":
        artifact_compilation = {
            "status": "invalid",
            "error_type": artifacts.get("error_type"),
            "error": artifacts.get("error"),
        }
    elif artifacts.get("compiled"):
        artifact_compilation = {
            "status": "ok",
            "target": artifacts.get("prefill", {}).get("target"),
        }
    else:
        artifact_compilation = {"status": "not_compared"}

    if artifacts.get("status") == "not_requested":
        artifact_execution = {"status": "not_requested"}
    elif dinoml.get("status") == "ok":
        artifact_execution = {
            "status": "ok",
            "execution_mode": dinoml.get("execution_mode"),
        }
    elif dinoml.get("status") in {"blocked", "not_run"}:
        artifact_execution = {
            "status": dinoml.get("status"),
            "error_type": dinoml.get("error_type"),
            "error": dinoml.get("error"),
            "reason": dinoml.get("reason"),
        }
    else:
        artifact_execution = {"status": "not_compared"}

    prefill_parity = compare_logits_outputs(
        dinoml,
        transformers,
        dinoml_key="_prefill_logits",
        transformers_key="_prefill_logits",
        compare_name="prefill_logits",
    )
    first_decode_parity = compare_logits_outputs(
        dinoml,
        transformers,
        dinoml_key="_first_decode_logits",
        transformers_key="_first_decode_logits",
        compare_name="first_decode_step_logits",
    )
    final_generated_output_parity = compare_generated_outputs(dinoml, transformers)
    acceptance_met = all(
        item.get("status") == "ok"
        for item in (
            artifact_compilation,
            artifact_execution,
            prefill_parity,
            first_decode_parity,
            final_generated_output_parity,
        )
    )
    return {
        "artifact_compilation": artifact_compilation,
        "artifact_execution": artifact_execution,
        "prefill_parity": prefill_parity,
        "first_decode_step_parity": first_decode_parity,
        "final_generated_output_parity": final_generated_output_parity,
        "acceptance_met": acceptance_met,
    }


def _probe_arrays_path(output_json: Path) -> Path:
    return output_json.with_suffix(".probes.npz")


def save_probe_arrays(summary: dict[str, Any], output_json: Path | None) -> dict[str, Any]:
    if output_json is None:
        return summary
    arrays: dict[str, np.ndarray] = {}
    for prefix in ("dinoml", "transformers"):
        section = summary.get(prefix)
        if not isinstance(section, dict):
            continue
        for key in ("_prefill_logits", "_first_decode_logits"):
            value = section.get(key)
            if value is not None:
                arrays[f"{prefix}{key}"] = np.asarray(value, dtype=np.float32)
    if not arrays:
        return summary
    probe_path = _probe_arrays_path(output_json)
    np.savez_compressed(probe_path, **arrays)
    summary.setdefault("probe_arrays", {})["path"] = str(probe_path)
    return summary


def load_probe_arrays(summary_path: Path) -> dict[str, np.ndarray]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    probe_path = Path((summary.get("probe_arrays") or {}).get("path", ""))
    if not probe_path.is_absolute() and not probe_path.is_file():
        probe_path = summary_path.parent / probe_path
    if not probe_path.is_file():
        raise FileNotFoundError(f"Probe array bundle not found for {summary_path}: {probe_path}")
    with np.load(probe_path) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def strip_internal_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_internal_payload(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, list):
        return [strip_internal_payload(item) for item in value]
    if isinstance(value, tuple):
        return [strip_internal_payload(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def compare_saved_runs(dinoml_json: Path, transformers_json: Path) -> dict[str, Any]:
    dinoml_summary = json.loads(dinoml_json.read_text(encoding="utf-8"))
    transformers_summary = json.loads(transformers_json.read_text(encoding="utf-8"))
    dinoml_arrays = load_probe_arrays(dinoml_json)
    transformers_arrays = load_probe_arrays(transformers_json)
    dinoml = dict(dinoml_summary.get("dinoml") or {})
    transformers = dict(transformers_summary.get("transformers") or {})
    dinoml["_prefill_logits"] = dinoml_arrays.get("dinoml_prefill_logits")
    dinoml["_first_decode_logits"] = dinoml_arrays.get("dinoml_first_decode_logits")
    transformers["_prefill_logits"] = transformers_arrays.get("transformers_prefill_logits")
    transformers["_first_decode_logits"] = transformers_arrays.get("transformers_first_decode_logits")
    verification = build_verification_summary(
        artifacts=dict(dinoml_summary.get("artifacts") or {}),
        dinoml=dinoml,
        transformers=transformers,
    )
    return {
        "status": "ok",
        "dinoml_json": str(dinoml_json),
        "transformers_json": str(transformers_json),
        "artifacts": dinoml_summary.get("artifacts"),
        "dinoml": strip_internal_payload(dinoml),
        "transformers": strip_internal_payload(transformers),
        "verification": verification,
    }


def probe_checkpoint_loader(config, snapshot: Path) -> dict[str, Any]:
    required = ["model.norm.weight", "visual.merger.ln_q.weight"]
    weights = load_qwen2_5_vl_weights(config=config, snapshot=snapshot, required_names=required)
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in weights.items()}


def torch_dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def run_transformers_reference(
    *,
    snapshot: Path,
    processor,
    processed_torch,
    dtype: str,
    device_name: str,
    max_new_tokens: int,
    stop_token_ids: tuple[int, ...],
) -> dict[str, Any]:
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false; ROCm PyTorch should expose ROCm devices through the cuda API")
    device = torch.device(device_name)
    model_kwargs = {
        "torch_dtype": torch_dtype(dtype),
        "local_files_only": True,
    }
    signature = inspect.signature(Qwen2_5_VLForConditionalGeneration.from_pretrained)
    if "dtype" in signature.parameters:
        model_kwargs["dtype"] = model_kwargs.pop("torch_dtype")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(snapshot, **model_kwargs).to(device)
    model.eval()
    inputs = {name: value.to(device) for name, value in processed_torch.items() if name in TRANSFORMERS_REFERENCE_INPUTS}
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_logits": True,
    }
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = int(pad_token_id)
    if stop_token_ids:
        generation_kwargs["eos_token_id"] = list(stop_token_ids) if len(stop_token_ids) > 1 else int(stop_token_ids[0])
    with torch.inference_mode():
        started = time.perf_counter()
        prefill = model(**inputs, use_cache=True, logits_to_keep=1)
        prefill_logits = prefill.logits.detach().float().cpu().numpy()
        generated = model.generate(
            **inputs,
            **generation_kwargs,
        )
        generation_time_ms = (time.perf_counter() - started) * 1000.0
    prompt_len = int(inputs["input_ids"].shape[1])
    generated_ids_tensor = generated.sequences[:, prompt_len:]
    generated_ids = [int(value) for value in generated_ids_tensor[0].detach().cpu().tolist()]
    first_decode_logits = None
    first_decode_probe: dict[str, Any] = {"status": "not_run", "reason": "generation_stopped_after_prefill"}
    raw_logits = tuple(getattr(generated, "logits", ()) or ())
    if len(raw_logits) >= 2 and generated_ids:
        first_decode_logits = raw_logits[1].detach().float().cpu().numpy()
        if first_decode_logits.ndim == 2:
            first_decode_logits = first_decode_logits[:, None, :]
        first_decode_probe = {
            "status": "ok",
            "input_token_id": int(generated_ids[0]),
            "argmax_token_id": int(torch.argmax(raw_logits[1][0, :]).item()),
            "logits_shape": list(first_decode_logits.shape),
        }
    decoded = processor.batch_decode(
        np.asarray([generated_ids], dtype=np.int64),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return {
        "status": "ok",
        "device": str(device),
        "generated_token_count": len(generated_ids),
        "generated_ids": generated_ids,
        "text": decoded[0] if decoded else "",
        "generation_time_ms": generation_time_ms,
        "prefill_output_preview": {
            "logits_shape": list(prefill.logits.shape),
            "argmax_token_id": int(np.argmax(prefill_logits.reshape(-1))),
        },
        "first_decode_output_preview": first_decode_probe,
        "_prefill_logits": prefill_logits,
        "_first_decode_logits": first_decode_logits if first_decode_probe.get("status") == "ok" else None,
    }


def array_shapes(values: dict[str, np.ndarray]) -> dict[str, list[int]]:
    return {name: [int(dim) for dim in value.shape] for name, value in values.items()}


def write_summary(summary: dict[str, Any], output_json: Path | None) -> None:
    summary = save_probe_arrays(summary, output_json)
    payload = json.dumps(strip_internal_payload(summary), indent=2, sort_keys=True)
    if output_json is not None:
        output_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)


def release_gpu_memory_before_transformers_reference() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def main() -> int:
    args = parse_args()
    if args.compare_json is not None:
        summary = compare_saved_runs(Path(args.compare_json[0]), Path(args.compare_json[1]))
        write_summary(summary, args.output_json)
        return 0
    summary: dict[str, Any] = {
        "snapshot": str(args.snapshot),
        "image": str(args.image),
        "prompt": args.prompt,
        "dtype": args.dtype,
        "max_new_tokens": int(args.max_new_tokens),
    }
    config = load_qwen2_5_vl_config(snapshot=args.snapshot, dtype=args.dtype)
    summary["config"] = {
        "hidden_size": int(config.text_config.hidden_size),
        "num_hidden_layers": int(config.text_config.num_hidden_layers),
        "num_attention_heads": int(config.text_config.num_attention_heads),
        "num_key_value_heads": int(config.text_config.num_key_value_heads),
        "head_dim": int(config.text_config.head_dim),
        "vision_patch_dim": int(config.vision_config.patch_dim),
        "vision_out_hidden_size": int(config.vision_config.out_hidden_size),
    }
    summary["checkpoint_probe"] = probe_checkpoint_loader(config, args.snapshot)
    processor, processed_torch, processed_numpy, image_size, source_image_size, processor_image_size = load_processor_and_inputs(
        args.snapshot,
        args.image,
        args.prompt,
        longest_side=args.longest_side,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    prefill_inputs, full_inputs, prefill_len, max_cache_len, rope_deltas = build_pipeline_inputs(
        config,
        processed_numpy,
        args.max_new_tokens,
    )
    summary["image"] = {
        "source_size": list(source_image_size),
        "processed_size": list(image_size),
        "processor_size": processor_image_size,
    }
    summary["processor_outputs"] = array_shapes({name: np.asarray(value) for name, value in processed_numpy.items()})
    summary["dinoml_contract"] = {
        "prefill_inputs": array_shapes(prefill_inputs),
        "full_inputs": array_shapes(full_inputs),
        "prefill_len": int(prefill_len),
        "max_cache_len": int(max_cache_len),
        "rope_deltas": np.asarray(rope_deltas).tolist(),
        "cache_spec_shape": list(cache_spec_for_config(config, max_cache_len).shape()),
    }
    summary["stop_token_ids"] = list(load_stop_token_ids(args.snapshot, processor))

    if args.prefill_artifact is not None or args.decode_artifact is not None:
        if args.prefill_artifact is None or args.decode_artifact is None:
            raise ValueError("--prefill-artifact and --decode-artifact must be provided together")
        try:
            artifact_modes = validate_artifacts(prefill_artifact=args.prefill_artifact, decode_artifact=args.decode_artifact)
        except Exception as exc:  # noqa: BLE001 - preserve validation blocker in verification output.
            summary["artifacts"] = {
                "status": "invalid",
                "requested": True,
                "compiled": False,
                "executed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            summary["dinoml"] = {"status": "not_run", "reason": "artifacts_invalid"}
            summary["transformers"] = {"status": "not_requested"}
            summary["verification"] = build_verification_summary(
                artifacts=summary["artifacts"],
                dinoml=summary["dinoml"],
                transformers=summary["transformers"],
            )
            write_summary(summary, args.output_json)
            return 2
        summary["artifacts"] = artifact_modes
        try:
            summary["dinoml"] = run_artifacts(
                prefill_artifact=args.prefill_artifact,
                decode_artifact=args.decode_artifact,
                artifact_modes=artifact_modes,
                processor=processor,
                prefill_inputs=prefill_inputs,
                full_inputs=full_inputs,
                config=config,
                prefill_len=prefill_len,
                max_cache_len=max_cache_len,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=tuple(summary["stop_token_ids"]),
                warmup=args.warmup,
                iterations=args.iterations,
            )
            summary["artifacts"]["executed"] = True
            summary["artifacts"]["execution_mode"] = summary["dinoml"].get("execution_mode")
        except Exception as exc:  # noqa: BLE001 - preserve artifact runtime blocker in verification output.
            summary["dinoml"] = {
                "status": "blocked",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            summary["artifacts"]["executed"] = False
            summary["transformers"] = {"status": "not_requested"}
            summary["verification"] = build_verification_summary(
                artifacts=summary["artifacts"],
                dinoml=summary["dinoml"],
                transformers=summary["transformers"],
            )
            write_summary(summary, args.output_json)
            return 2
    else:
        summary["artifacts"] = {"status": "not_requested", "requested": False, "compiled": False, "executed": False}
        summary["dinoml"] = {"status": "not_requested"}

    if args.compare_transformers:
        release_gpu_memory_before_transformers_reference()
        try:
            summary["transformers"] = run_transformers_reference(
                snapshot=args.snapshot,
                processor=processor,
                processed_torch=processed_torch,
                dtype=args.dtype,
                device_name=args.transformers_device,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=tuple(summary["stop_token_ids"]),
            )
        except Exception as exc:  # noqa: BLE001 - keep blocker details in the verification payload.
            summary["transformers"] = {
                "status": "blocked",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            summary["verification"] = build_verification_summary(
                artifacts=summary["artifacts"],
                dinoml=summary["dinoml"],
                transformers=summary["transformers"],
            )
            write_summary(summary, args.output_json)
            return 2
    else:
        summary["transformers"] = {"status": "not_requested"}

    summary["parity"] = compare_generated_outputs(summary["dinoml"], summary["transformers"])
    summary["verification"] = build_verification_summary(
        artifacts=summary["artifacts"],
        dinoml=summary["dinoml"],
        transformers=summary["transformers"],
    )

    write_summary(summary, args.output_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
