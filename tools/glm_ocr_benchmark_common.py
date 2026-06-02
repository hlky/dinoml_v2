from __future__ import annotations

from pathlib import Path

from PIL import Image
from transformers.image_utils import SizeDict


DEFAULT_SNAPSHOT = Path(
    r"C:\Users\user\.cache\huggingface\hub\models--zai-org--GLM-OCR\snapshots\ca5d8b3e287e52589e37c28385d9655ee4372f9d"
)
DEFAULT_IMAGE = Path(
    r"K:\Mulder24B\data\mkultra\raw\DOC_0000017352\DOC_0000017352\0000017352_0001.TIF"
)
DEFAULT_PROMPT = "Perform OCR on this document image. Return the text only."


def configure_processor_image_size(processor, min_pixels: int | None, max_pixels: int | None) -> dict[str, int] | None:
    if min_pixels is None and max_pixels is None:
        return None
    current = processor.image_processor.size
    shortest_edge = min_pixels if min_pixels is not None else int(current.shortest_edge)
    longest_edge = max_pixels if max_pixels is not None else int(current.longest_edge)
    processor.image_processor.size = SizeDict(shortest_edge=shortest_edge, longest_edge=longest_edge)
    return {"shortest_edge": shortest_edge, "longest_edge": longest_edge}


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
