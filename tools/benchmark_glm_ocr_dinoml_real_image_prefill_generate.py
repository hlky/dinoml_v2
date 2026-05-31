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
    GlmOcrForConditionalGenerationImagePrefill,
    glm_ocr_config_from_transformers_dict,
    glm_ocr_rope_index,
    glm_ocr_text_rope_embeddings,
    glm_ocr_vision_position_ids,
    glm_ocr_vision_rope_embeddings,
    glm_ocr_weights_from_safetensors_file,
)


DEFAULT_SNAPSHOT = Path(
    r"C:\Users\user\.cache\huggingface\hub\models--zai-org--GLM-OCR\snapshots\ca5d8b3e287e52589e37c28385d9655ee4372f9d"
)
DEFAULT_IMAGE = Path(
    r"K:\Mulder24B\data\mkultra\raw\DOC_0000017352\DOC_0000017352\0000017352_0001.TIF"
)
DEFAULT_PROMPT = "Perform OCR on this document image. Return the text only."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark real-image GLM-OCR greedy generation through a fixed DinoML full-prefill graph."
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--target", choices=("rocm", "cuda", "cpu"), default="rocm")
    parser.add_argument("--arch", default=None)
    parser.add_argument("--min-pixels", type=int, default=784)
    parser.add_argument("--max-pixels", type=int, default=50176)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--execution-plan", type=Path)
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


def build_static_inputs(config, processed: dict[str, np.ndarray], max_new_tokens: int) -> tuple[dict[str, np.ndarray], int, int]:
    prompt_ids = np.asarray(processed["input_ids"], dtype=np.int64)
    mm_token_type_ids = np.asarray(processed["mm_token_type_ids"], dtype=np.int64)
    image_grid_thw = np.asarray(processed["image_grid_thw"], dtype=np.int64)
    pixel_values = np.asarray(processed["pixel_values"], dtype=np.float32)
    prompt_len = int(prompt_ids.shape[1])
    max_seq_len = prompt_len + max_new_tokens
    pad_id = 0
    input_ids = np.full((1, max_seq_len), pad_id, dtype=np.int64)
    input_ids[:, :prompt_len] = prompt_ids
    mm_full = np.zeros((1, max_seq_len), dtype=np.int64)
    mm_full[:, :prompt_len] = mm_token_type_ids
    text_position_ids, _ = glm_ocr_rope_index(
        input_ids,
        mm_full,
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
    attention_mask = np.triu(
        np.full((config.text_config.num_attention_heads, max_seq_len, max_seq_len), -1.0e4, dtype=np.float32),
        k=1,
    )
    image_positions = np.flatnonzero(mm_full[0, :prompt_len] == 1)
    if image_positions.size == 0:
        image_positions = np.flatnonzero(prompt_ids[0] == config.image_token_id)
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
            "attention_mask": _float_input(attention_mask, config.text_config.dtype),
        },
        prompt_len,
        int(image_positions[0]),
    )


def build_spec(config, weights: dict[str, np.ndarray], inputs: dict[str, np.ndarray], image_token_start: int):
    return dml.trace(
        GlmOcrForConditionalGenerationImagePrefill(
            config,
            weights,
            image_token_start=image_token_start,
            logits_to_keep=0,
        ),
        inputs={
            "input_ids": dml.TensorSpec(list(inputs["input_ids"].shape), "int64"),
            "pixel_values": dml.TensorSpec(list(inputs["pixel_values"].shape), config.vision_config.dtype),
            "vision_cos": dml.TensorSpec(list(inputs["vision_cos"].shape), config.vision_config.dtype),
            "vision_sin": dml.TensorSpec(list(inputs["vision_sin"].shape), config.vision_config.dtype),
            "text_cos": dml.TensorSpec(list(inputs["text_cos"].shape), config.text_config.dtype),
            "text_sin": dml.TensorSpec(list(inputs["text_sin"].shape), config.text_config.dtype),
            "attention_mask": dml.TensorSpec(list(inputs["attention_mask"].shape), config.text_config.dtype),
        },
        name=f"glm_ocr_real_image_prefill_generate_s{inputs['input_ids'].shape[1]}_p{inputs['pixel_values'].shape[0]}",
    )


def ensure_artifact(args: argparse.Namespace, config, inputs, image_token_start: int) -> Path:
    artifact = args.artifact
    if artifact is None:
        artifact = Path("build") / (
            f"glm_ocr_real_image_prefill_generate_s{inputs['input_ids'].shape[1]}"
            f"_p{inputs['pixel_values'].shape[0]}_{args.dtype}_{args.target}.dinoml"
        )
    if artifact.exists() and not args.force_compile:
        return artifact
    weights = glm_ocr_weights_from_safetensors_file(
        args.snapshot / "model.safetensors",
        config,
        dtype=config.text_config.dtype,
    )
    spec = build_spec(config, weights, inputs, image_token_start)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    dml.compile(
        spec,
        target=dml.Target(args.target, arch=args.arch),
        output=artifact,
        execution_plan=args.execution_plan,
        profile=args.profile_compile,
        profile_iterations=args.profile_iterations,
        profile_repeats=args.profile_repeats,
    )
    return artifact


def generate_once(session, base_inputs: dict[str, np.ndarray], prompt_len: int, max_new_tokens: int, eos_token_id: int):
    inputs = {name: np.array(value, copy=True) for name, value in base_inputs.items()}
    generated: list[int] = []
    run_times_ms: list[float] = []
    for step in range(max_new_tokens):
        current_position = prompt_len + step - 1
        started = time.perf_counter()
        outputs = session.run_numpy(inputs)
        run_times_ms.append((time.perf_counter() - started) * 1000.0)
        logits = outputs["logits"][0, current_position, :]
        next_id = int(np.argmax(logits))
        generated.append(next_id)
        if next_id == eos_token_id:
            break
        if prompt_len + step < inputs["input_ids"].shape[1]:
            inputs["input_ids"][0, prompt_len + step] = next_id
    return generated, run_times_ms


def _float_input(values: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bfloat16":
        if values.dtype == np.uint16:
            return np.ascontiguousarray(values)
        return array_to_storage(values.astype(np.float32, copy=False), "bfloat16")
    return values.astype(dtype, copy=False)


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
    inputs, prompt_len, image_token_start = build_static_inputs(config, processed, args.max_new_tokens)
    started_compile = time.perf_counter()
    artifact = ensure_artifact(args, config, inputs, image_token_start)
    compile_seconds = time.perf_counter() - started_compile

    rt_module = runtime.load(artifact, load_constants=True)
    session = None
    try:
        session = rt_module.create_session()
        eos_token_id = int(processor.tokenizer.eos_token_id)
        for _ in range(args.warmup):
            generate_once(session, inputs, prompt_len, args.max_new_tokens, eos_token_id)
        times_ms: list[float] = []
        token_run_times_ms: list[list[float]] = []
        generated_ids: list[int] = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            generated_ids, per_token_times = generate_once(session, inputs, prompt_len, args.max_new_tokens, eos_token_id)
            times_ms.append((time.perf_counter() - started) * 1000.0)
            token_run_times_ms.append(per_token_times)
    finally:
        if session is not None:
            session.close()
        rt_module.close()

    text = processor.post_process_image_text_to_text(np.asarray([generated_ids], dtype=np.int64), skip_special_tokens=True)[0]
    payload = {
        "benchmark": "dinoml_glm_ocr_real_image_full_prefill_generate",
        "artifact": str(artifact),
        "snapshot": str(args.snapshot),
        "image": str(args.image),
        "image_size": list(image_size),
        "prompt": args.prompt,
        "transformers_version": transformers_version,
        "transformers_file": transformers_file,
        "dtype": args.dtype,
        "target": args.target,
        "arch": args.arch,
        "processor_image_size": processor_image_size,
        "max_new_tokens": args.max_new_tokens,
        "prompt_len": prompt_len,
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
