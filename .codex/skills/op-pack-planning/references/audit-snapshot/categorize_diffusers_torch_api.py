from __future__ import annotations

import argparse
import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import Callable


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def group(names: list[str], categorizer: Callable[[str], str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for name in names:
        grouped[categorizer(name)].append(name)
    return dict(sorted(grouped.items()))


def emit_grouped(lines: list[str], title: str, grouped: dict[str, list[str]]) -> None:
    total = sum(len(names) for names in grouped.values())
    lines.extend([f"## {title} ({total})", ""])
    for category, names in grouped.items():
        lines.extend([f"### {category} ({len(names)})", ""])
        lines.append(", ".join(f"`{name}`" for name in names) or "-")
        lines.append("")


def write_report(
    output: Path,
    title: str,
    source: Path,
    note: str,
    torch_tensor: list[str],
    nn_modules: list[str],
    functional: list[str],
    categorizers,
) -> None:
    lines = [
        f"# {title}",
        "",
        f"Source: `{source}`",
        "",
        note,
        "",
    ]
    emit_grouped(lines, "torch and torch.Tensor functions", group(torch_tensor, categorizers.category_for_torch_tensor))
    emit_grouped(lines, "torch.nn modules", group(nn_modules, categorizers.category_for_nn_module))
    emit_grouped(lines, "torch.nn.functional functions", group(functional, categorizers.category_for_functional))
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Categorize used and unused public PyTorch APIs for Diffusers.")
    parser.add_argument(
        "diffusers_dir",
        nargs="?",
        default=r"X:\H\diffusers",
        help="Diffusers repo root, src/diffusers, or diffusers package directory.",
    )
    parser.add_argument("--used-output", type=Path, default=Path("diffusers_torch_api_used_categories.md"))
    parser.add_argument("--unused-output", type=Path, default=Path("diffusers_torch_api_unused_categories.md"))
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    diffusers_api = load_module("diffusers_torch_api", here / "diffusers_torch_api.py")
    analyzer = load_module("torch_api_by_model_family", here / "torch_api_by_model_family.py")
    categorizers = load_module("categorize_unused_torch_api", here / "categorize_unused_torch_api.py")

    public_api = analyzer.load_public_api()
    public_function_api = analyzer.load_public_function_api()
    package = diffusers_api.resolve_diffusers_package(args.diffusers_dir)
    data = diffusers_api.scan(package, public_api)

    used_torch_tensor = sorted(diffusers_api.aggregate_counts(data, ("torch", "torch.Tensor"), normalize_inplace=True))
    used_nn = sorted(diffusers_api.aggregate_counts(data, ("torch.nn",)))
    used_functional = sorted(diffusers_api.aggregate_counts(data, ("torch.nn.functional",), normalize_inplace=True))

    public_torch_tensor = analyzer.public_names(public_function_api, ("torch", "torch.Tensor"), normalize_inplace=True)
    public_nn = analyzer.public_nn_modules()
    public_functional = analyzer.public_names(public_function_api, ("torch.nn.functional",), normalize_inplace=True)

    unused_torch_tensor = sorted(public_torch_tensor - set(used_torch_tensor))
    unused_nn = sorted(public_nn - set(used_nn))
    unused_functional = sorted(public_functional - set(used_functional))

    write_report(
        args.used_output,
        "Categorized Public Torch APIs Used by Diffusers Model Components",
        package,
        "Used means present in the selected Diffusers model component files. Function names are normalized so in-place suffixes such as `_` are folded into the base name.",
        used_torch_tensor,
        used_nn,
        used_functional,
        categorizers,
    )
    write_report(
        args.unused_output,
        "Categorized Public Torch APIs Not Used by Diffusers Model Components",
        package,
        "Unused means absent from the selected Diffusers model component files. Function names are normalized so in-place suffixes such as `_` are folded into the base name.",
        unused_torch_tensor,
        unused_nn,
        unused_functional,
        categorizers,
    )


if __name__ == "__main__":
    main()
