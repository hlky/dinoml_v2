from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from dinoml import runtime
from dinoml.models.glm_ocr import (
    glm_ocr_rope_index,
    glm_ocr_text_rope_embeddings,
    glm_ocr_vision_position_ids,
    glm_ocr_vision_rope_embeddings,
)
from dinoml.models.glm_ocr.workflow_common import (
    float_input,
    load_glm_ocr_config,
)
from dinoml.models.kv_cache import (
    StaticKvCacheSpec,
    seed_static_kv_cache,
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
    parser = argparse.ArgumentParser(
        description="Run and benchmark a GLM-OCR image-prefill + decode pipeline."
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prefill-artifact", type=Path, required=True)
    parser.add_argument("--decode-artifact", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1, help="Whole-pipeline warmup iterations.")
    parser.add_argument("--iterations", type=int, default=3, help="Whole-pipeline timed iterations.")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument(
        "--longest-side",
        type=int,
        default=None,
        help="Resize the source image to this longest edge before processing.",
    )
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
    import torch
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
    try:
        processed_torch = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"return_mm_token_type_ids": True},
        )
    except TypeError:
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        processed_torch = processor(text=[text], images=[image], return_tensors="pt", return_mm_token_type_ids=True)
    processed_numpy = {name: value.detach().cpu().numpy() for name, value in processed_torch.items()}
    return processor, processed_torch, processed_numpy, image.size, source_image_size, processor_image_size


def build_pipeline_inputs(
    config,
    processed: dict[str, np.ndarray],
    max_new_tokens: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int, int]:
    input_ids = np.asarray(processed["input_ids"], dtype=np.int64)
    mm_token_type_ids = np.asarray(processed["mm_token_type_ids"], dtype=np.int64)
    image_grid_thw = np.asarray(processed["image_grid_thw"], dtype=np.int64)
    pixel_values = np.asarray(processed["pixel_values"], dtype=np.float32)
    text_attention_mask = processed.get("attention_mask")
    if text_attention_mask is not None:
        text_attention_mask = np.asarray(text_attention_mask, dtype=np.int64)
    prefill_len = int(input_ids.shape[1])
    max_cache_len = prefill_len + max_new_tokens

    prefill_position_ids, rope_deltas = glm_ocr_rope_index(
        input_ids,
        mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=text_attention_mask,
        spatial_merge_size=config.vision_config.spatial_merge_size,
    )
    decode_text_positions = np.arange(prefill_len, max_cache_len, dtype=np.int64).reshape(1, 1, max_new_tokens)
    decode_text_positions = np.broadcast_to(
        decode_text_positions,
        (3, input_ids.shape[0], max_new_tokens),
    ).copy()
    decode_text_positions += np.asarray(rope_deltas, dtype=np.int64).reshape(1, input_ids.shape[0], 1)
    text_position_ids = np.concatenate([prefill_position_ids, decode_text_positions], axis=2)
    text_cos, text_sin = glm_ocr_text_rope_embeddings(
        text_position_ids,
        config.text_config,
        dtype=config.text_config.dtype,
    )
    vision_position_ids = glm_ocr_vision_position_ids(
        image_grid_thw,
        config.vision_config.spatial_merge_size,
    )
    vision_cos, vision_sin = glm_ocr_vision_rope_embeddings(
        vision_position_ids,
        head_dim=config.vision_config.head_dim,
    )
    full_inputs = {
        "input_ids": input_ids,
        "pixel_values": float_input(pixel_values, config.vision_config.dtype),
        "vision_cos": float_input(vision_cos, "float32"),
        "vision_sin": float_input(vision_sin, "float32"),
        "text_cos": float_input(text_cos, config.text_config.dtype),
        "text_sin": float_input(text_sin, config.text_config.dtype),
        "attention_mask": prefill_attention_mask(
            config,
            processed.get("attention_mask"),
            prompt_len=prefill_len,
        ),
    }
    prefill_inputs = {
        "input_ids": full_inputs["input_ids"],
        "pixel_values": full_inputs["pixel_values"],
        "vision_cos": full_inputs["vision_cos"],
        "vision_sin": full_inputs["vision_sin"],
        "text_cos": np.ascontiguousarray(full_inputs["text_cos"][:, :prefill_len, :]),
        "text_sin": np.ascontiguousarray(full_inputs["text_sin"][:, :prefill_len, :]),
        "attention_mask": full_inputs["attention_mask"],
    }
    return prefill_inputs, full_inputs, prefill_len, max_cache_len


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


def validate_artifacts(*, prefill_artifact: Path, decode_artifact: Path) -> dict[str, Any]:
    prefill_metadata = load_artifact_metadata(prefill_artifact)
    decode_metadata = load_artifact_metadata(decode_artifact)
    prefill_inputs = {entry["name"] for entry in prefill_metadata.get("inputs", [])}
    prefill_outputs = {entry["name"] for entry in prefill_metadata.get("outputs", [])}
    decode_inputs = {entry["name"] for entry in decode_metadata.get("inputs", [])}
    decode_outputs = {entry["name"] for entry in decode_metadata.get("outputs", [])}

    required_prefill_inputs = {
        "input_ids",
        "pixel_values",
        "vision_cos",
        "vision_sin",
        "text_cos",
        "text_sin",
        "attention_mask",
    }
    missing_prefill = sorted(required_prefill_inputs - prefill_inputs)
    if missing_prefill:
        raise ValueError(
            f"Prefill artifact {prefill_artifact} is missing expected inputs: {', '.join(missing_prefill)}"
        )
    if not any(name.startswith("present_key_") for name in prefill_outputs):
        raise ValueError(f"Prefill artifact {prefill_artifact} does not expose cache outputs")

    required_decode_inputs = {"input_ids", "cos", "sin"}
    missing_decode = sorted(required_decode_inputs - decode_inputs)
    if missing_decode:
        raise ValueError(
            f"Decode artifact {decode_artifact} is missing expected inputs: {', '.join(missing_decode)}"
        )
    if not any(name.startswith("past_key_") for name in decode_inputs):
        raise ValueError(f"Decode artifact {decode_artifact} does not expose decode cache inputs")
    if any(name.startswith("new_key_") for name in decode_outputs):
        if "cache_seqlens" not in decode_inputs:
            raise ValueError(
                f"Decode artifact {decode_artifact} exposes static KV cache updates but is missing cache_seqlens"
            )
        return {
            "decode_mode": "static",
            "use_decode_attention_mask": "attention_mask" in decode_inputs,
        }
    if any(name.startswith("present_key_") for name in decode_outputs):
        return {
            "decode_mode": "dynamic",
            "use_decode_attention_mask": "attention_mask" in decode_inputs,
        }
    raise ValueError(
        f"Decode artifact {decode_artifact} must expose either present_key_* outputs or new_key_* outputs"
    )


def prefill_attention_mask(config, attention_mask: object | None, *, prompt_len: int) -> np.ndarray:
    mask_fill_value = float(getattr(config.text_config, "mask_fill_value", -1.0e4))
    # Prefill runs flash_attention_bias(..., causal=True), so the bias only needs
    # to carry extra masking such as padding; causality comes from the kernel.
    mask = np.zeros(
        (config.text_config.num_attention_heads, prompt_len, prompt_len),
        dtype=np.float32,
    )
    if attention_mask is not None:
        keep = np.asarray(attention_mask, dtype=bool)
        if keep.ndim != 2 or keep.shape[0] != 1 or keep.shape[1] < prompt_len:
            raise ValueError(f"attention_mask must have shape [1, >= {prompt_len}], got {keep.shape}")
        mask[:, :, ~keep[0, :prompt_len]] = mask_fill_value
    return float_input(mask, config.text_config.dtype)


def decode_attention_mask(config, total_len: int, *, valid_past_len: int) -> np.ndarray:
    mask_fill_value = float(getattr(config.text_config, "mask_fill_value", -1.0e4))
    mask = np.full((config.text_config.num_attention_heads, 1, total_len), mask_fill_value, dtype=np.float32)
    mask[:, :, : int(valid_past_len) + 1] = 0.0
    return float_input(mask, config.text_config.dtype)


def seed_dynamic_decode_cache(prefill_outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        f"past_key_{layer_idx}": np.ascontiguousarray(prefill_outputs[f"present_key_{layer_idx}"])
        for layer_idx in range(len([name for name in prefill_outputs if name.startswith("present_key_")]))
    } | {
        f"past_value_{layer_idx}": np.ascontiguousarray(prefill_outputs[f"present_value_{layer_idx}"])
        for layer_idx in range(len([name for name in prefill_outputs if name.startswith("present_value_")]))
    }


def update_dynamic_decode_cache(step_outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        f"past_key_{layer_idx}": np.ascontiguousarray(step_outputs[f"present_key_{layer_idx}"])
        for layer_idx in range(len([name for name in step_outputs if name.startswith("present_key_")]))
    } | {
        f"past_value_{layer_idx}": np.ascontiguousarray(step_outputs[f"present_value_{layer_idx}"])
        for layer_idx in range(len([name for name in step_outputs if name.startswith("present_value_")]))
    }


def slice_position(values: np.ndarray, position: int) -> np.ndarray:
    return np.ascontiguousarray(values[:, int(position) : int(position) + 1, :])


def decode_step_inputs(
    *,
    full_inputs: dict[str, np.ndarray],
    config,
    cache: dict[str, np.ndarray],
    next_id: int,
    position: int,
    decode_mode: str,
    use_attention_mask: bool,
) -> dict[str, np.ndarray]:
    inputs = {
        "input_ids": np.asarray([[next_id]], dtype=np.int64),
        "cos": slice_position(full_inputs["text_cos"], position),
        "sin": slice_position(full_inputs["text_sin"], position),
    }
    if decode_mode == "static":
        total_len = int(position) + 1
        inputs["cache_seqlens"] = np.asarray([position], dtype=np.int32)
        inputs.update(
            {
                name: np.ascontiguousarray(value[:, :, :total_len, :])
                for name, value in cache.items()
            }
        )
    else:
        total_len = int(position) + 1
        inputs.update({name: np.ascontiguousarray(value) for name, value in cache.items()})
    if use_attention_mask:
        inputs["attention_mask"] = decode_attention_mask(config, total_len, valid_past_len=position)
    return inputs


def run_pipeline_once(
    prefill_session,
    decode_session,
    *,
    prefill_inputs: dict[str, np.ndarray],
    full_inputs: dict[str, np.ndarray],
    config,
    decode_mode: str,
    use_decode_attention_mask: bool,
    prefill_len: int,
    max_cache_len: int,
    max_new_tokens: int,
    stop_token_ids: tuple[int, ...],
) -> tuple[list[int], dict[str, Any], list[float]]:
    prefill_outputs = prefill_session.run_numpy(prefill_inputs)
    next_id = int(np.argmax(prefill_outputs["logits"][0, 0, :]))
    generated_ids: list[int] = []
    decode_times_ms: list[float] = []
    if decode_mode == "static":
        cache_spec = cache_spec_for_config(config, max_cache_len)
        cache = seed_static_kv_cache(prefill_outputs, cache_spec, cache_len=prefill_len)
    else:
        cache = seed_dynamic_decode_cache(prefill_outputs)
        cache_spec = None

    for step in range(max_new_tokens):
        generated_ids.append(next_id)
        if next_id in stop_token_ids or step == max_new_tokens - 1:
            break
        position = prefill_len + step
        step_inputs = decode_step_inputs(
            full_inputs=full_inputs,
            config=config,
            cache=cache,
            next_id=next_id,
            position=position,
            decode_mode=decode_mode,
            use_attention_mask=use_decode_attention_mask,
        )
        started = time.perf_counter()
        step_outputs = decode_session.run_numpy(step_inputs)
        decode_times_ms.append((time.perf_counter() - started) * 1000.0)
        if decode_mode == "static":
            write_static_kv_cache_update(cache, step_outputs, cache_spec, position=position)
        else:
            cache = update_dynamic_decode_cache(step_outputs)
        next_id = int(np.argmax(step_outputs["logits"][0, 0, :]))
    return generated_ids, prefill_outputs, decode_times_ms


def torch_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


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
    prefill_session,
    decode_session,
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
    prefill_session.set_stream(stream)
    decode_session.set_stream(stream)
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
        "input_ids_t": torch.empty((1, 1), dtype=torch.int64, device=device),
        "cos_t": torch.empty((1, 1, int(config.text_config.head_dim)), dtype=torch_dtype(config.text_config.dtype), device=device),
        "sin_t": torch.empty((1, 1, int(config.text_config.head_dim)), dtype=torch_dtype(config.text_config.dtype), device=device),
        "cache_seqlens_t": torch.empty((1,), dtype=torch.int32, device=device),
        "attention_mask_t": attention_mask_t,
    }


def run_pipeline_once_device(
    prefill_session,
    decode_session,
    *,
    config,
    device_ctx: dict[str, Any],
    use_decode_attention_mask: bool,
    prefill_len: int,
    max_cache_len: int,
    max_new_tokens: int,
    stop_token_ids: tuple[int, ...],
) -> tuple[list[int], dict[str, Any], list[float]]:
    import torch

    device = device_ctx["device"]
    stream = device_ctx["stream"]
    prefill_inputs_t = device_ctx["prefill_inputs_t"]
    full_inputs_t = device_ctx["full_inputs_t"]
    prefill_shapes = device_ctx["prefill_shapes"]
    prefill_outputs_t = device_ctx["prefill_outputs_t"]
    decode_output_shapes_map = device_ctx["decode_output_shapes_map"]
    decode_outputs_t = device_ctx["decode_outputs_t"]
    input_ids_t = device_ctx["input_ids_t"]
    cos_t = device_ctx["cos_t"]
    sin_t = device_ctx["sin_t"]
    cache_seqlens_t = device_ctx["cache_seqlens_t"]
    attention_mask_t = device_ctx["attention_mask_t"]
    decode_mode = str(device_ctx["decode_mode"])
    mask_fill_value = float(getattr(config.text_config, "mask_fill_value", -1.0e4))
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
                cache_t[f"past_key_{layer_idx}"][:, :, :prefill_len, :].copy_(prefill_outputs_t[f"present_key_{layer_idx}"])
                cache_t[f"past_value_{layer_idx}"][:, :, :prefill_len, :].copy_(prefill_outputs_t[f"present_value_{layer_idx}"])
        else:
            cache_t = {
                f"past_key_{layer_idx}": prefill_outputs_t[f"present_key_{layer_idx}"].clone()
                for layer_idx in range(int(config.text_config.num_hidden_layers))
            } | {
                f"past_value_{layer_idx}": prefill_outputs_t[f"present_value_{layer_idx}"].clone()
                for layer_idx in range(int(config.text_config.num_hidden_layers))
            }

        next_id = int(torch.argmax(prefill_outputs_t["logits"][0, 0, :]).item())
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
            if decode_mode == "static":
                cache_seqlens_t.fill_(position)
                decode_inputs_t["cache_seqlens"] = cache_seqlens_t
                decode_input_shapes["cache_seqlens"] = (1,)
            for layer_idx in range(int(config.text_config.num_hidden_layers)):
                cache_len = total_len if decode_mode == "static" else position
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
                decode_input_shapes["attention_mask"] = (
                    int(config.text_config.num_attention_heads),
                    1,
                    total_len,
                )
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
    return generated_ids, {"logits_shape": list(prefill_outputs_t["logits"].shape)}, []


def benchmark_transformers(
    *,
    snapshot: Path,
    processor,
    processed_torch,
    prompt_len: int,
    max_new_tokens: int,
    device_name: str,
    dtype_name: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    import torch
    from transformers import GlmOcrForConditionalGeneration

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device requested for Transformers benchmark, but torch.cuda.is_available() is false.")
    dtype = torch_dtype(dtype_name)
    try:
        model = GlmOcrForConditionalGeneration.from_pretrained(snapshot, dtype=dtype)
    except TypeError:
        model = GlmOcrForConditionalGeneration.from_pretrained(snapshot, torch_dtype=dtype)
    model = model.to(device).eval()
    inputs = {name: value.to(device) for name, value in processed_torch.items()}

    def generate_once():
        return model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )

    times_ms: list[float] = []
    final_ids = None
    with torch.inference_mode():
        for _ in range(warmup):
            _ = generate_once()
            if device.type == "cuda":
                torch.cuda.synchronize()
        for _ in range(iterations):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                final_ids = generate_once()
                end.record()
                torch.cuda.synchronize()
                elapsed_ms = float(start.elapsed_time(end))
            else:
                started = time.perf_counter()
                final_ids = generate_once()
                elapsed_ms = (time.perf_counter() - started) * 1000.0
            times_ms.append(elapsed_ms)
    assert final_ids is not None
    generated_ids = final_ids[:, prompt_len:]
    text = processor.post_process_image_text_to_text(generated_ids, skip_special_tokens=True)[0]
    return {
        "device": str(device),
        "dtype": dtype_name,
        "warmup": warmup,
        "iterations": iterations,
        "generated_tokens": int(generated_ids.shape[1]),
        "times_ms": times_ms,
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.fmean(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "text": text,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    from transformers import __file__ as transformers_file
    from transformers import __version__ as transformers_version

    processor, processed_torch, processed_numpy, image_size, source_image_size, processor_image_size = (
        load_processor_and_inputs(
            args.snapshot,
            args.image,
            args.prompt,
            longest_side=args.longest_side,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
    )
    config = load_glm_ocr_config(snapshot=args.snapshot, dtype=args.dtype)
    prefill_inputs, full_inputs, prefill_len, max_cache_len = build_pipeline_inputs(
        config,
        processed_numpy,
        args.max_new_tokens,
    )
    artifact_modes = validate_artifacts(
        prefill_artifact=args.prefill_artifact,
        decode_artifact=args.decode_artifact,
    )

    prefill_module = runtime.load(args.prefill_artifact, load_constants=True)
    decode_module = runtime.load(args.decode_artifact, load_constants=True)
    execution_mode = "numpy"
    device = None
    device_ctx = None
    prefill_session = None
    decode_session = None
    try:
        prefill_session = prefill_module.create_session()
        decode_session = decode_module.create_session()
        if can_use_device_pointers(prefill_module) and can_use_device_pointers(decode_module):
            try:
                import torch

                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    execution_mode = "device_pointers"
                    device_ctx = prepare_device_pipeline(
                        prefill_session=prefill_session,
                        decode_session=decode_session,
                    prefill_inputs=prefill_inputs,
                    full_inputs=full_inputs,
                    config=config,
                    decode_mode=str(artifact_modes["decode_mode"]),
                    prefill_len=prefill_len,
                    max_cache_len=max_cache_len,
                    device=device,
                    )
            except ImportError:
                pass
        stop_token_ids = load_stop_token_ids(args.snapshot, processor)

        pipeline_times_ms: list[float] = []
        decode_step_runs_ms: list[list[float]] = []
        generated_ids: list[int] = []
        prefill_output_preview: dict[str, Any] = {}
        for _ in range(args.warmup):
            if execution_mode == "device_pointers":
                run_pipeline_once_device(
                    prefill_session,
                    decode_session,
                    config=config,
                    device_ctx=device_ctx,
                    use_decode_attention_mask=bool(artifact_modes["use_decode_attention_mask"]),
                    prefill_len=prefill_len,
                    max_cache_len=max_cache_len,
                    max_new_tokens=args.max_new_tokens,
                    stop_token_ids=stop_token_ids,
                )
            else:
                run_pipeline_once(
                    prefill_session,
                    decode_session,
                    prefill_inputs=prefill_inputs,
                    full_inputs=full_inputs,
                    config=config,
                    decode_mode=str(artifact_modes["decode_mode"]),
                    use_decode_attention_mask=bool(artifact_modes["use_decode_attention_mask"]),
                    prefill_len=prefill_len,
                    max_cache_len=max_cache_len,
                    max_new_tokens=args.max_new_tokens,
                    stop_token_ids=stop_token_ids,
                )
        for _ in range(args.iterations):
            started = time.perf_counter()
            if execution_mode == "device_pointers":
                generated_ids, prefill_outputs, decode_times = run_pipeline_once_device(
                    prefill_session,
                    decode_session,
                    config=config,
                    device_ctx=device_ctx,
                    use_decode_attention_mask=bool(artifact_modes["use_decode_attention_mask"]),
                    prefill_len=prefill_len,
                    max_cache_len=max_cache_len,
                    max_new_tokens=args.max_new_tokens,
                    stop_token_ids=stop_token_ids,
                )
            else:
                generated_ids, prefill_outputs, decode_times = run_pipeline_once(
                    prefill_session,
                    decode_session,
                    prefill_inputs=prefill_inputs,
                    full_inputs=full_inputs,
                    config=config,
                    decode_mode=str(artifact_modes["decode_mode"]),
                    use_decode_attention_mask=bool(artifact_modes["use_decode_attention_mask"]),
                    prefill_len=prefill_len,
                    max_cache_len=max_cache_len,
                    max_new_tokens=args.max_new_tokens,
                    stop_token_ids=stop_token_ids,
                )
            pipeline_times_ms.append((time.perf_counter() - started) * 1000.0)
            decode_step_runs_ms.append(decode_times)
            prefill_output_preview = (
                {"logits_shape": list(prefill_outputs["logits"].shape)}
                if "logits" in prefill_outputs
                else prefill_outputs
            )
    finally:
        if decode_session is not None:
            decode_session.close()
        if prefill_session is not None:
            prefill_session.close()
        decode_module.close()
        prefill_module.close()

    text = processor.post_process_image_text_to_text(
        np.asarray([generated_ids], dtype=np.int64),
        skip_special_tokens=True,
    )[0]
    payload: dict[str, Any] = {
        "benchmark": "dinoml_glm_ocr_pipeline",
        "snapshot": str(args.snapshot),
        "image": str(args.image),
        "source_image_size": list(source_image_size),
        "image_size": list(image_size),
        "longest_side": args.longest_side,
        "prompt": args.prompt,
        "dtype": args.dtype,
        "decode_mode": str(artifact_modes["decode_mode"]),
        "use_decode_attention_mask": bool(artifact_modes["use_decode_attention_mask"]),
        "max_new_tokens": args.max_new_tokens,
        "prefill_len": prefill_len,
        "max_cache_len": max_cache_len,
        "input_shapes": {name: list(value.shape) for name, value in full_inputs.items()},
        "processor_image_size": processor_image_size,
        "prefill_artifact": str(args.prefill_artifact),
        "decode_artifact": str(args.decode_artifact),
        "execution_mode": execution_mode,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "pipeline_times_ms": pipeline_times_ms,
        "pipeline_median_ms": statistics.median(pipeline_times_ms),
        "pipeline_mean_ms": statistics.fmean(pipeline_times_ms),
        "pipeline_min_ms": min(pipeline_times_ms),
        "pipeline_max_ms": max(pipeline_times_ms),
        "decode_step_runs_ms": decode_step_runs_ms,
        "generated_tokens": len(generated_ids),
        "generated_ids": generated_ids,
        "text": text,
        "prefill_output_preview": prefill_output_preview,
        "transformers_version": transformers_version,
        "transformers_file": transformers_file,
    }
    if args.compare_transformers:
        payload["transformers_benchmark"] = benchmark_transformers(
            snapshot=args.snapshot,
            processor=processor,
            processed_torch=processed_torch,
            prompt_len=prefill_len,
            max_new_tokens=args.max_new_tokens,
            device_name=args.transformers_device,
            dtype_name=args.dtype,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        payload["text_matches_transformers"] = (
            payload["transformers_benchmark"]["text"] == payload["text"]
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
