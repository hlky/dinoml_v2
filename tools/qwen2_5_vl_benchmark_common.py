from __future__ import annotations

from pathlib import Path

from PIL import Image


DEFAULT_SNAPSHOT = Path(r"G:\checkpoints\Qwen\Qwen2.5-VL-3B-Instruct")
DEFAULT_IMAGE = Path(r"K:\0000017352_0001.jpg")
DEFAULT_PROMPT = "Describe the document image. Return the recognized text only."


def configure_processor_image_size(processor, min_pixels: int | None, max_pixels: int | None) -> dict[str, int] | None:
    if min_pixels is None and max_pixels is None:
        return None
    image_processor = processor.image_processor
    if min_pixels is not None:
        image_processor.min_pixels = int(min_pixels)
    if max_pixels is not None:
        image_processor.max_pixels = int(max_pixels)
    return {
        "min_pixels": int(getattr(image_processor, "min_pixels", min_pixels or 0)),
        "max_pixels": int(getattr(image_processor, "max_pixels", max_pixels or 0)),
    }


def resize_image_longest_side(image: Image.Image, longest_side: int | None) -> Image.Image:
    if longest_side is None:
        return image
    if longest_side <= 0:
        raise ValueError("--longest-side must be positive")
    width, height = image.size
    current_longest = max(width, height)
    if current_longest == longest_side:
        return image
    scale = float(longest_side) / float(current_longest)
    resized_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return image.resize(resized_size, resampling)


def open_rgb_image(path: Path, *, longest_side: int | None = None) -> tuple[Image.Image, tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    source_size = image.size
    return resize_image_longest_side(image, longest_side), source_size
