from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from dinoml.models.glm_ocr import (
    glm_ocr_config_from_transformers_dict,
    glm_ocr_rope_index,
)


DEFAULT_SNAPSHOT = Path(
    r"C:\Users\user\.cache\huggingface\hub\models--zai-org--GLM-OCR\snapshots\ca5d8b3e287e52589e37c28385d9655ee4372f9d"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GLM-OCR single-image prefill in Transformers.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--grid-thw", default="1,8,8")
    parser.add_argument("--text-tail-len", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_model(args: argparse.Namespace, dtype: torch.dtype, device: torch.device):
    from transformers import GlmOcrForConditionalGeneration

    kwargs = {"dtype": dtype}
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    try:
        model = GlmOcrForConditionalGeneration.from_pretrained(args.snapshot, **kwargs)
    except TypeError:
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = dtype
        model = GlmOcrForConditionalGeneration.from_pretrained(args.snapshot, **kwargs)
    return model.to(device).eval()


def build_inputs(snapshot: Path, grid_thw: tuple[int, int, int], text_tail_len: int, device: torch.device, dtype: torch.dtype):
    config_payload = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    config = glm_ocr_config_from_transformers_dict(config_payload, dtype="float32")
    merge = config.vision_config.spatial_merge_size
    grid_t, grid_h, grid_w = grid_thw
    if grid_h % merge or grid_w % merge:
        raise ValueError("--grid-thw height and width must be divisible by the model merge size")
    image_feature_count = grid_t * grid_h * grid_w // (merge * merge)
    patch_count = grid_t * grid_h * grid_w
    seq_len = 1 + image_feature_count + text_tail_len
    input_ids_np = np.concatenate(
        [
            np.asarray([42], dtype=np.int64),
            np.full((image_feature_count,), config.image_token_id, dtype=np.int64),
            np.full((text_tail_len,), 43, dtype=np.int64),
        ]
    ).reshape(1, seq_len)
    mm_token_type_ids_np = np.concatenate(
        [
            np.zeros((1,), dtype=np.int64),
            np.ones((image_feature_count,), dtype=np.int64),
            np.zeros((text_tail_len,), dtype=np.int64),
        ]
    ).reshape(1, seq_len)
    image_grid_thw_np = np.asarray([grid_thw], dtype=np.int64)
    position_ids_np, _ = glm_ocr_rope_index(
        input_ids_np,
        mm_token_type_ids_np,
        image_grid_thw=image_grid_thw_np,
        spatial_merge_size=merge,
    )
    rng = np.random.default_rng(20260530)
    pixel_values_np = rng.normal(0.0, 0.2, (patch_count, config.vision_config.patch_dim)).astype(np.float32)
    return {
        "input_ids": torch.from_numpy(input_ids_np).to(device=device),
        "pixel_values": torch.from_numpy(pixel_values_np).to(device=device, dtype=dtype),
        "image_grid_thw": torch.from_numpy(image_grid_thw_np).to(device=device),
        "mm_token_type_ids": torch.from_numpy(mm_token_type_ids_np).to(device=device),
        "position_ids": torch.from_numpy(position_ids_np).to(device=device),
        "logits_to_keep": 1,
    }, config


def main() -> None:
    args = parse_args()
    grid_thw = tuple(int(part) for part in args.grid_thw.split(","))
    if len(grid_thw) != 3:
        raise ValueError("--grid-thw must be formatted as T,H,W")
    device = torch.device(args.device)
    dtype = torch_dtype(args.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device requested, but torch.cuda.is_available() is false.")

    from transformers import __file__ as transformers_file
    from transformers import __version__ as transformers_version

    started = time.perf_counter()
    model = load_model(args, dtype, device)
    load_seconds = time.perf_counter() - started
    inputs, config = build_inputs(args.snapshot, grid_thw, args.text_tail_len, device, dtype)

    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(**inputs).logits
            if device.type == "cuda":
                torch.cuda.synchronize()

        times_ms: list[float] = []
        for _ in range(args.iterations):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                logits = model(**inputs).logits
                end.record()
                torch.cuda.synchronize()
                elapsed_ms = start.elapsed_time(end)
            else:
                started_iter = time.perf_counter()
                logits = model(**inputs).logits
                elapsed_ms = (time.perf_counter() - started_iter) * 1000.0
            if tuple(logits.shape) != (1, 1, config.text_config.vocab_size):
                raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
            times_ms.append(float(elapsed_ms))

    payload = {
        "benchmark": "transformers_glm_ocr_image_prefill",
        "snapshot": str(args.snapshot),
        "transformers_version": transformers_version,
        "transformers_file": transformers_file,
        "torch_version": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "grid_thw": list(grid_thw),
        "text_tail_len": args.text_tail_len,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "load_seconds": load_seconds,
        "times_ms": times_ms,
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.fmean(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }
    print(json.dumps(payload, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
