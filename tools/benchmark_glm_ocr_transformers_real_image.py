from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
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
    parser = argparse.ArgumentParser(description="Benchmark real-image GLM-OCR text generation in Transformers.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--longest-side", type=int, default=None, help="Resize the source image to this longest edge before processing.")
    parser.add_argument("--min-pixels", type=int, help="Override processor shortest_edge pixel-count bound.")
    parser.add_argument("--max-pixels", type=int, help="Override processor longest_edge pixel-count bound.")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_model(snapshot: Path, dtype: torch.dtype, device: torch.device):
    from transformers import GlmOcrForConditionalGeneration

    kwargs = {"dtype": dtype}
    try:
        model = GlmOcrForConditionalGeneration.from_pretrained(snapshot, **kwargs)
    except TypeError:
        model = GlmOcrForConditionalGeneration.from_pretrained(snapshot, torch_dtype=dtype)
    return model.to(device).eval()


def build_inputs(processor, image_path: Path, prompt: str, device: torch.device, *, longest_side: int | None = None):
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
    return {name: value.to(device) for name, value in inputs.items()}, image.size, source_image_size


def generated_suffix(output_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    return output_ids[:, prompt_len:]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch_dtype(args.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device requested, but torch.cuda.is_available() is false.")

    from transformers import AutoProcessor
    from transformers import __file__ as transformers_file
    from transformers import __version__ as transformers_version

    processor = AutoProcessor.from_pretrained(args.snapshot)
    processor_image_size = configure_processor_image_size(processor, args.min_pixels, args.max_pixels)
    started = time.perf_counter()
    model = load_model(args.snapshot, dtype, device)
    load_seconds = time.perf_counter() - started
    inputs, image_size, source_image_size = build_inputs(
        processor,
        args.image,
        args.prompt,
        device,
        longest_side=args.longest_side,
    )
    prompt_len = int(inputs["input_ids"].shape[1])

    def generate_once() -> torch.Tensor:
        return model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
        )

    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = generate_once()
            if device.type == "cuda":
                torch.cuda.synchronize()

        times_ms: list[float] = []
        final_ids = None
        for _ in range(args.iterations):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                final_ids = generate_once()
                end.record()
                torch.cuda.synchronize()
                elapsed_ms = start.elapsed_time(end)
            else:
                started_iter = time.perf_counter()
                final_ids = generate_once()
                elapsed_ms = (time.perf_counter() - started_iter) * 1000.0
            times_ms.append(float(elapsed_ms))

    assert final_ids is not None
    new_ids = generated_suffix(final_ids, prompt_len)
    text = processor.post_process_image_text_to_text(new_ids, skip_special_tokens=True)[0]
    payload = {
        "benchmark": "transformers_glm_ocr_real_image_generate",
        "snapshot": str(args.snapshot),
        "image": str(args.image),
        "source_image_size": list(source_image_size),
        "image_size": list(image_size),
        "longest_side": args.longest_side,
        "prompt": args.prompt,
        "transformers_version": transformers_version,
        "transformers_file": transformers_file,
        "torch_version": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dtype": args.dtype,
        "processor_image_size": processor_image_size,
        "max_new_tokens": args.max_new_tokens,
        "prompt_len": prompt_len,
        "generated_tokens": int(new_ids.shape[1]),
        "input_shapes": {name: list(value.shape) for name, value in inputs.items()},
        "warmup": args.warmup,
        "iterations": args.iterations,
        "load_seconds": load_seconds,
        "times_ms": times_ms,
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
