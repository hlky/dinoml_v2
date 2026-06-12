from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


API_KEYS = ("torch", "torch.Tensor", "torch.nn", "torch.nn.functional")

TOP_LEVEL_MODEL_FILES = (
    "activations.py",
    "adapter.py",
    "attention.py",
    "downsampling.py",
    "embeddings.py",
    "normalization.py",
    "resnet.py",
    "upsampling.py",
    "vq_model.py",
)
SUBDIR_RULES = {
    "autoencoders": (),
    "controlnets": ("_flax",),
    "transformers": (),
    "unets": ("_flax",),
}


def load_transformers_analyzer():
    path = Path(__file__).with_name("torch_api_by_model_family.py")
    spec = importlib.util.spec_from_file_location("torch_api_by_model_family", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load analyzer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_diffusers_package(path_arg: str | None) -> Path:
    if path_arg:
        root = Path(path_arg).expanduser().resolve()
        candidates = (
            root / "src" / "diffusers",
            root / "diffusers",
            root,
        )
        for candidate in candidates:
            if (candidate / "models").is_dir():
                return candidate
        raise SystemExit(f"Could not find a diffusers/models directory under {root}")

    try:
        diffusers = importlib.import_module("diffusers")
    except ImportError as exc:
        raise SystemExit(
            "No Diffusers source directory was provided and the diffusers package "
            "is not importable from site-packages."
        ) from exc

    package_dir = Path(diffusers.__file__).resolve().parent
    if not (package_dir / "models").is_dir():
        raise SystemExit(f"Imported diffusers, but no models directory exists at {package_dir}")
    return package_dir


def include_subdir_file(path: Path, excluded_name_parts: tuple[str, ...]) -> bool:
    if path.name == "__init__.py" or path.suffix != ".py":
        return False
    return not any(part in path.stem for part in excluded_name_parts)


def component_files(diffusers_package: Path) -> dict[str, Path]:
    models_dir = diffusers_package / "models"
    files: dict[str, Path] = {}

    for filename in TOP_LEVEL_MODEL_FILES:
        path = models_dir / filename
        if path.is_file():
            files[path.stem] = path

    for dirname, excluded_name_parts in SUBDIR_RULES.items():
        directory = models_dir / dirname
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.py")):
            if include_subdir_file(path, excluded_name_parts):
                files[f"{dirname}/{path.stem}"] = path

    return dict(sorted(files.items()))


def scan(diffusers_package: Path, public_api: dict[str, set[str]] | None = None) -> dict[str, dict[str, Any]]:
    analyzer = load_transformers_analyzer()
    if public_api is None:
        public_api = analyzer.load_public_api()

    data: dict[str, dict[str, Any]] = {}
    for component, path in component_files(diffusers_package).items():
        used = analyzer.scan_file(path, public_api)
        data[component] = {
            "files": [str(path.relative_to(diffusers_package))],
            **{key: sorted(used[key]) for key in API_KEYS},
        }
    return data


def write_json(data: dict[str, dict[str, Any]], output: Path | None) -> None:
    text = json.dumps(data, indent=2, sort_keys=True)
    if output:
        output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def rows(data: dict[str, dict[str, Any]]) -> Iterable[dict[str, str]]:
    for component, entry in data.items():
        for key in API_KEYS:
            for name in entry[key]:
                yield {"component": component, "api_group": key, "name": name}


def write_csv(data: dict[str, dict[str, Any]], output: Path | None) -> None:
    if output:
        handle = output.open("w", newline="", encoding="utf-8")
        close = True
    else:
        handle = sys.stdout
        close = False

    try:
        writer = csv.DictWriter(handle, fieldnames=("component", "api_group", "name"))
        writer.writeheader()
        writer.writerows(rows(data))
    finally:
        if close:
            handle.close()


def write_markdown(data: dict[str, dict[str, Any]], output: Path | None) -> None:
    lines: list[str] = []
    for component, entry in data.items():
        lines.append(f"## {component}")
        for key in API_KEYS:
            values = ", ".join(f"`{name}`" for name in entry[key]) or "-"
            lines.append(f"- `{key}`: {values}")
        lines.append("")

    text = "\n".join(lines).rstrip() + "\n"
    if output:
        output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def aggregate_counts(
    data: dict[str, dict[str, Any]],
    keys: tuple[str, ...],
    *,
    normalize_inplace: bool = False,
) -> dict[str, int]:
    analyzer = load_transformers_analyzer()
    counts: dict[str, int] = defaultdict(int)
    for entry in data.values():
        names = set()
        for key in keys:
            for name in entry[key]:
                names.add(analyzer.normalize_name(name, normalize_inplace=normalize_inplace))
        for name in names:
            counts[name] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def markdown_count_table(label: str, name_header: str, counts: dict[str, int]) -> list[str]:
    lines = [
        f"## {label}",
        "",
        f"| {name_header} | count |",
        "| --- | ---: |",
    ]
    lines.extend(f"| `{name}` | {count} |" for name, count in counts.items())
    lines.append("")
    return lines


def markdown_alphabetical_list(label: str, counts: dict[str, int]) -> list[str]:
    lines = [f"## {label}", ""]
    lines.extend(f"- `{name}`" for name in sorted(counts))
    lines.append("")
    return lines


def write_aggregate_markdown(data: dict[str, dict[str, Any]], output: Path | None) -> None:
    torch_tensor_counts = aggregate_counts(data, ("torch", "torch.Tensor"), normalize_inplace=True)
    nn_counts = aggregate_counts(data, ("torch.nn",))
    functional_counts = aggregate_counts(data, ("torch.nn.functional",), normalize_inplace=True)

    lines = ["# Torch API Usage by Diffusers Model Components", ""]
    lines.extend(markdown_count_table("torch and torch.Tensor", "function", torch_tensor_counts))
    lines.extend(markdown_count_table("torch.nn", "module", nn_counts))
    lines.extend(markdown_count_table("torch.nn.functional", "function", functional_counts))
    lines.extend(markdown_alphabetical_list("Alphabetical torch and torch.Tensor Functions", torch_tensor_counts))
    lines.extend(markdown_alphabetical_list("Alphabetical torch.nn Modules", nn_counts))
    lines.extend(markdown_alphabetical_list("Alphabetical torch.nn.functional Functions", functional_counts))

    text = "\n".join(lines).rstrip() + "\n"
    if output:
        output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def write_unused_markdown(
    data: dict[str, dict[str, Any]],
    public_api: dict[str, set[str]],
    output: Path | None,
) -> None:
    analyzer = load_transformers_analyzer()
    public_function_api = analyzer.load_public_function_api()
    sections = (
        (
            "Unused torch and torch.Tensor Functions",
            analyzer.public_names(public_function_api, ("torch", "torch.Tensor"), normalize_inplace=True),
            set(aggregate_counts(data, ("torch", "torch.Tensor"), normalize_inplace=True)),
        ),
        (
            "Unused torch.nn Modules",
            analyzer.public_nn_modules(),
            set(aggregate_counts(data, ("torch.nn",))),
        ),
        (
            "Unused torch.nn.functional Functions",
            analyzer.public_names(public_function_api, ("torch.nn.functional",), normalize_inplace=True),
            set(aggregate_counts(data, ("torch.nn.functional",), normalize_inplace=True)),
        ),
    )

    lines = ["# Public Torch APIs Not Used by Diffusers Model Components", ""]
    for label, public, used in sections:
        unused = sorted(public - used)
        lines.extend([f"## {label} ({len(unused)})", ""])
        lines.extend(f"- `{name}`" for name in unused)
        lines.append("")

    text = "\n".join(lines).rstrip() + "\n"
    if output:
        output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List torch, torch.Tensor, torch.nn, and torch.nn.functional calls "
            "used by selected Diffusers model components."
        )
    )
    parser.add_argument(
        "diffusers_dir",
        nargs="?",
        help=(
            "Path to a Diffusers repo root, src/diffusers, or diffusers package "
            "directory. Defaults to site-packages diffusers."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("json", "csv", "markdown", "aggregate-markdown", "unused-markdown"),
        default="json",
        help="Output format. Default: json.",
    )
    parser.add_argument("--output", type=Path, help="Write output to this file instead of stdout.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyzer = load_transformers_analyzer()
    public_api = analyzer.load_public_api()
    diffusers_package = resolve_diffusers_package(args.diffusers_dir)
    data = scan(diffusers_package, public_api)

    if args.format == "json":
        write_json(data, args.output)
    elif args.format == "csv":
        write_csv(data, args.output)
    elif args.format == "markdown":
        write_markdown(data, args.output)
    elif args.format == "aggregate-markdown":
        write_aggregate_markdown(data, args.output)
    else:
        write_unused_markdown(data, public_api, args.output)


if __name__ == "__main__":
    main()
