from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


DEFAULT_SNAPSHOT = Path(
    r"C:\Users\user\.cache\huggingface\hub\models--zai-org--GLM-OCR\snapshots\ca5d8b3e287e52589e37c28385d9655ee4372f9d"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GLM-OCR one-token decode in Transformers.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--past-len", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
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
        kwargs = {"torch_dtype": dtype}
        model = GlmOcrForConditionalGeneration.from_pretrained(snapshot, **kwargs)
    return model.to(device).eval()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch_dtype(args.dtype)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device requested, but torch.cuda.is_available() is false.")
    if args.past_len < 1:
        raise ValueError("--past-len must be positive")
    if args.batch < 1:
        raise ValueError("--batch must be positive")

    from transformers import __file__ as transformers_file
    from transformers import __version__ as transformers_version

    started = time.perf_counter()
    model = load_model(args.snapshot, dtype, device)
    load_seconds = time.perf_counter() - started

    input_ids = torch.full((args.batch, args.past_len), 42, dtype=torch.long, device=device)
    attention_mask = torch.ones((args.batch, args.past_len), dtype=torch.long, device=device)
    position_ids = torch.arange(args.past_len, dtype=torch.long, device=device)
    position_ids = position_ids.view(1, 1, -1).expand(3, args.batch, -1)

    decode_ids = torch.full((args.batch, 1), 43, dtype=torch.long, device=device)
    decode_attention_mask = torch.ones((args.batch, args.past_len + 1), dtype=torch.long, device=device)
    decode_position_ids = torch.full((3, args.batch, 1), args.past_len, dtype=torch.long, device=device)

    with torch.inference_mode():
        prefill = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = prefill.past_key_values

        for _ in range(args.warmup):
            _ = model(
                input_ids=decode_ids,
                attention_mask=decode_attention_mask,
                position_ids=decode_position_ids,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            ).logits
            if device.type == "cuda":
                torch.cuda.synchronize()
            cache.crop(args.past_len)

        times_ms: list[float] = []
        for _ in range(args.iterations):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                logits = model(
                    input_ids=decode_ids,
                    attention_mask=decode_attention_mask,
                    position_ids=decode_position_ids,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                ).logits
                end.record()
                torch.cuda.synchronize()
                elapsed_ms = start.elapsed_time(end)
            else:
                started_iter = time.perf_counter()
                logits = model(
                    input_ids=decode_ids,
                    attention_mask=decode_attention_mask,
                    position_ids=decode_position_ids,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                ).logits
                elapsed_ms = (time.perf_counter() - started_iter) * 1000.0

            if tuple(logits.shape) != (args.batch, 1, model.config.text_config.vocab_size):
                raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
            times_ms.append(float(elapsed_ms))
            cache.crop(args.past_len)

    payload = {
        "benchmark": "transformers_glm_ocr_decode",
        "snapshot": str(args.snapshot),
        "transformers_version": transformers_version,
        "transformers_file": transformers_file,
        "torch_version": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dtype": args.dtype,
        "batch": args.batch,
        "past_len": args.past_len,
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
