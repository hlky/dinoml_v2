from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from transformers.image_utils import SizeDict

import dinoml as dml
from dinoml import runtime
from dinoml.ir import array_to_storage
from dinoml.models.glm_ocr import (
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
    seed_static_kv_cache,
    static_kv_cache_input_specs,
    write_static_kv_cache_update,
)


DEFAULT_SNAPSHOT = Path(
    r"C:\Users\user\.cache\huggingface\hub\models--zai-org--GLM-OCR\snapshots\ca5d8b3e287e52589e37c28385d9655ee4372f9d"
)
DEFAULT_IMAGE = Path(
    r"K:\Mulder24B\data\mkultra\raw\DOC_0000017352\DOC_0000017352\0000017352_0001.preview.png"
)
DEFAULT_PROMPT = "Perform OCR on this document image. Return the text only."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark real-image GLM-OCR generation with DinoML KV cache.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--target", choices=("rocm", "cuda", "cpu"), default="rocm")
    parser.add_argument("--arch", default=None)
    parser.add_argument("--min-pixels", type=int, default=784)
    parser.add_argument("--max-pixels", type=int, default=1298237)
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


def configure_processor_image_size(processor, min_pixels: int | None, max_pixels: int | None) -> dict[str, int] | None:
    if min_pixels is None and max_pixels is None:
        return None
    current = processor.image_processor.size
    shortest_edge = min_pixels if min_pixels is not None else int(current.shortest_edge)
    longest_edge = max_pixels if max_pixels is not None else int(current.longest_edge)
    processor.image_processor.size = SizeDict(shortest_edge=shortest_edge, longest_edge=longest_edge)
    return {"shortest_edge": shortest_edge, "longest_edge": longest_edge}


def processor_inputs(processor, image_path: Path, prompt: str):
    image = Image.open(image_path).convert("RGB")
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
            return_mm_token_type_ids=True,
        )
    except TypeError:
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text], images=[image], return_tensors="pt", return_mm_token_type_ids=True)
    return {name: value.detach().cpu().numpy() for name, value in inputs.items()}, image.size


def build_inputs(config, processed: dict[str, np.ndarray], max_new_tokens: int):
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
        dtype=config.vision_config.dtype,
    )
    prefill_attention_mask = np.triu(
        np.full((config.text_config.num_attention_heads, prompt_len, prompt_len), -1.0e4, dtype=np.float32),
        k=1,
    )
    image_positions = np.flatnonzero(mm_token_type_ids[0] == 1)
    if image_positions.size == 0:
        image_positions = np.flatnonzero(input_ids[0] == config.image_token_id)
    if image_positions.size == 0:
        raise RuntimeError("processor inputs did not contain image placeholder tokens")
    return (
        {
            "input_ids": input_ids,
            "pixel_values": _float_input(pixel_values, config.vision_config.dtype),
            "vision_cos": _float_input(vision_cos, config.vision_config.dtype),
            "vision_sin": _float_input(vision_sin, config.vision_config.dtype),
            "text_cos": _float_input(text_cos, config.text_config.dtype),
            "text_sin": _float_input(text_sin, config.text_config.dtype),
            "attention_mask": _float_input(prefill_attention_mask, config.text_config.dtype),
        },
        prompt_len,
        max_cache_len,
        int(image_positions[0]),
    )


def build_prefill_spec(config, weights: dict[str, np.ndarray], inputs: dict[str, np.ndarray], image_token_start: int):
    return dml.trace(
        GlmOcrForConditionalGenerationImagePrefillWithCache(
            config,
            weights,
            image_token_start=image_token_start,
            logits_to_keep=1,
        ),
        inputs={
            "input_ids": dml.TensorSpec(list(inputs["input_ids"].shape), "int64"),
            "pixel_values": dml.TensorSpec(list(inputs["pixel_values"].shape), config.vision_config.dtype),
            "vision_cos": dml.TensorSpec(list(inputs["vision_cos"].shape), config.vision_config.dtype),
            "vision_sin": dml.TensorSpec(list(inputs["vision_sin"].shape), config.vision_config.dtype),
            "text_cos": dml.TensorSpec(
                [inputs["input_ids"].shape[0], inputs["input_ids"].shape[1], config.text_config.head_dim],
                config.text_config.dtype,
            ),
            "text_sin": dml.TensorSpec(
                [inputs["input_ids"].shape[0], inputs["input_ids"].shape[1], config.text_config.head_dim],
                config.text_config.dtype,
            ),
            "attention_mask": dml.TensorSpec(list(inputs["attention_mask"].shape), config.text_config.dtype),
        },
        name=f"glm_ocr_real_image_cached_prefill_s{inputs['input_ids'].shape[1]}_p{inputs['pixel_values'].shape[0]}",
    )


def build_decode_spec(
    config,
    weights: dict[str, np.ndarray],
    max_cache_len: int,
    *,
    use_flash_static_kv_cache: bool,
):
    cache_spec = cache_spec_for_config(config, max_cache_len)
    inputs = {
        "input_ids": dml.TensorSpec([1, 1], "int64"),
        "cos": dml.TensorSpec([1, 1, config.text_config.head_dim], config.text_config.dtype),
        "sin": dml.TensorSpec([1, 1, config.text_config.head_dim], config.text_config.dtype),
        "attention_mask": dml.TensorSpec(
            [config.text_config.num_attention_heads, 1, max_cache_len + 1],
            config.text_config.dtype,
        ),
        **static_kv_cache_input_specs(cache_spec),
    }
    if use_flash_static_kv_cache:
        inputs["cache_seqlens"] = dml.TensorSpec([1], "int32")
    return dml.trace(
        GlmOcrForConditionalGenerationDecodeStaticCache(config, weights),
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
    prefill_artifact = args.prefill_artifact
    if prefill_artifact is None:
        prefill_artifact = Path("build") / (
            f"glm_ocr_real_image_cached_prefill_s{inputs['input_ids'].shape[1]}"
            f"_p{inputs['pixel_values'].shape[0]}_{args.dtype}_{args.target}.dinoml"
        )
    decode_artifact = args.decode_artifact
    if decode_artifact is None:
        cache_suffix = "_flash_static_kv" if use_flash_static_kv_cache else ""
        decode_artifact = Path("build") / (
            f"glm_ocr_real_image_cached_decode_past{max_cache_len}{cache_suffix}_{args.dtype}_{args.target}.dinoml"
        )
    prefill_ready = _artifact_ready(prefill_artifact)
    decode_ready = _artifact_ready(decode_artifact)
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
) -> tuple[list[int], list[float]]:
    started = time.perf_counter()
    prefill_outputs = prefill_session.run_numpy(_prefill_run_inputs(prefill_inputs, prompt_len))
    run_times_ms = [(time.perf_counter() - started) * 1000.0]
    next_id = int(np.argmax(prefill_outputs["logits"][0, 0, :]))
    generated: list[int] = []
    cache_spec = cache_spec_for_config(config, max_cache_len)
    cache = seed_static_kv_cache(prefill_outputs, cache_spec, cache_len=prompt_len)

    for step in range(max_new_tokens):
        generated.append(next_id)
        if next_id == eos_token_id or step == max_new_tokens - 1:
            break
        position = prompt_len + step
        decode_inputs = {
            "input_ids": np.asarray([[next_id]], dtype=np.int64),
            "cos": _slice_position(prefill_inputs["text_cos"], position),
            "sin": _slice_position(prefill_inputs["text_sin"], position),
            "attention_mask": _decode_attention_mask(config, max_cache_len, valid_past_len=position),
            **cache,
        }
        if use_flash_static_kv_cache:
            decode_inputs["cache_seqlens"] = np.asarray([position], dtype=np.int32)
        started = time.perf_counter()
        decode_outputs = decode_session.run_numpy(decode_inputs)
        run_times_ms.append((time.perf_counter() - started) * 1000.0)
        write_static_kv_cache_update(cache, decode_outputs, cache_spec, position=position)
        next_id = int(np.argmax(decode_outputs["logits"][0, 0, :]))
    return generated, run_times_ms


def _prefill_run_inputs(inputs: dict[str, np.ndarray], prompt_len: int) -> dict[str, np.ndarray]:
    return {
        "input_ids": inputs["input_ids"],
        "pixel_values": inputs["pixel_values"],
        "vision_cos": inputs["vision_cos"],
        "vision_sin": inputs["vision_sin"],
        "text_cos": inputs["text_cos"][:, :prompt_len, :],
        "text_sin": inputs["text_sin"][:, :prompt_len, :],
        "attention_mask": inputs["attention_mask"],
    }


def _slice_position(values: np.ndarray, position: int) -> np.ndarray:
    return np.ascontiguousarray(values[:, int(position) : int(position) + 1, :])


def _decode_attention_mask(config, max_cache_len: int, *, valid_past_len: int) -> np.ndarray:
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
    return args.target in {"cuda", "rocm"} and config.text_config.dtype in {"float16", "bfloat16"}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    from transformers import AutoProcessor
    from transformers import __file__ as transformers_file
    from transformers import __version__ as transformers_version

    processor = AutoProcessor.from_pretrained(args.snapshot)
    processor_image_size = configure_processor_image_size(processor, args.min_pixels, args.max_pixels)
    processed, image_size = processor_inputs(processor, args.image, args.prompt)
    config = build_config(args.snapshot, args.dtype)
    inputs, prompt_len, max_cache_len, image_token_start = build_inputs(config, processed, args.max_new_tokens)
    use_flash_static_kv_cache = _use_flash_static_kv_cache(args, config)
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
            )
        times_ms: list[float] = []
        token_run_times_ms: list[list[float]] = []
        generated_ids: list[int] = []
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

    text = processor.post_process_image_text_to_text(np.asarray([generated_ids], dtype=np.int64), skip_special_tokens=True)[0]
    payload = {
        "benchmark": "dinoml_glm_ocr_real_image_cached_generate",
        "prefill_artifact": str(prefill_artifact),
        "decode_artifact": str(decode_artifact),
        "snapshot": str(args.snapshot),
        "image": str(args.image),
        "image_size": list(image_size),
        "prompt": args.prompt,
        "transformers_version": transformers_version,
        "transformers_file": transformers_file,
        "dtype": args.dtype,
        "target": args.target,
        "arch": args.arch,
        "use_flash_static_kv_cache": use_flash_static_kv_cache,
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
        "compile_seconds": compile_seconds,
        "times_ms": times_ms,
        "token_run_times_ms": token_run_times_ms,
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.fmean(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "text": text,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
