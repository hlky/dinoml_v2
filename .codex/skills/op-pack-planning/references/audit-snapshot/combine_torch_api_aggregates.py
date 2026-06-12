from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


TABLES = (
    ("torch and torch.Tensor", "function", ("torch", "torch.Tensor"), True),
    ("torch.nn", "module", ("torch.nn",), False),
    ("torch.nn.functional", "function", ("torch.nn.functional",), True),
)


def normalize_name(name: str, *, normalize_inplace: bool) -> str:
    if normalize_inplace and name.endswith("_"):
        return name[:-1]
    return name


def load_data(path: Path) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_counts(
    data: dict[str, dict[str, Any]],
    keys: tuple[str, ...],
    *,
    normalize_inplace: bool,
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for entry in data.values():
        names = set()
        for key in keys:
            for name in entry[key]:
                names.add(normalize_name(name, normalize_inplace=normalize_inplace))
        for name in names:
            counts[name] += 1
    return dict(counts)


def markdown_table(
    title: str,
    name_header: str,
    transformers_counts: dict[str, int],
    diffusers_counts: dict[str, int],
) -> list[str]:
    names = sorted(
        set(transformers_counts) | set(diffusers_counts),
        key=lambda name: (-(transformers_counts.get(name, 0) + diffusers_counts.get(name, 0)), name),
    )
    lines = [
        f"## {title}",
        "",
        f"| {name_header} | combined | transformers | diffusers |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in names:
        transformers_count = transformers_counts.get(name, 0)
        diffusers_count = diffusers_counts.get(name, 0)
        combined = transformers_count + diffusers_count
        lines.append(f"| `{name}` | {combined} | {transformers_count} | {diffusers_count} |")
    lines.append("")
    return lines


def write_report(
    transformers_data: dict[str, dict[str, Any]],
    diffusers_data: dict[str, dict[str, Any]],
    output: Path,
) -> None:
    lines = [
        "# Combined Torch API Usage in Transformers and Diffusers",
        "",
        "Counts are aggregate presence counts: each Transformers model family or Diffusers model component contributes at most one count per API name. In-place function names are normalized by removing a trailing `_`.",
        "",
        f"- Transformers units: {len(transformers_data)} model families",
        f"- Diffusers units: {len(diffusers_data)} model components",
        "",
    ]

    for title, name_header, keys, normalize_inplace in TABLES:
        transformers_counts = aggregate_counts(transformers_data, keys, normalize_inplace=normalize_inplace)
        diffusers_counts = aggregate_counts(diffusers_data, keys, normalize_inplace=normalize_inplace)
        lines.extend(markdown_table(title, name_header, transformers_counts, diffusers_counts))

    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine Transformers and Diffusers torch API aggregate counts.")
    parser.add_argument(
        "--transformers-json",
        type=Path,
        default=Path("torch_api_by_model_family.json"),
        help="Per-family Transformers JSON produced by torch_api_by_model_family.py.",
    )
    parser.add_argument(
        "--diffusers-json",
        type=Path,
        default=Path("diffusers_torch_api_by_component.json"),
        help="Per-component Diffusers JSON produced by diffusers_torch_api.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("combined_torch_api_aggregate_report.md"),
        help="Markdown output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_report(
        load_data(args.transformers_json),
        load_data(args.diffusers_json),
        args.output,
    )


if __name__ == "__main__":
    main()
