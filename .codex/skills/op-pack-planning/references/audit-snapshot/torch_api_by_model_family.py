from __future__ import annotations

import argparse
import ast
import csv
import inspect
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


API_KEYS = ("torch", "torch.Tensor", "torch.nn", "torch.nn.functional")
IGNORED_TENSOR_BASES = {
    "F",
    "T",
    "math",
    "nn",
    "np",
    "numpy",
    "os",
    "self",
    "sys",
    "torch",
}


def public_callables(obj: Any) -> set[str]:
    names: set[str] = set()
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value):
            names.add(name)
    return names


def public_nonclass_callables(obj: Any) -> set[str]:
    names: set[str] = set()
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value) and not inspect.isclass(value):
            names.add(name)
    return names


def public_nn_modules() -> set[str]:
    import torch.nn as nn

    names: set[str] = set()
    for name in dir(nn):
        if name.startswith("_"):
            continue
        try:
            value = getattr(nn, name)
        except Exception:
            continue
        if inspect.isclass(value) and issubclass(value, nn.Module):
            names.add(name)
    return names


def load_public_function_api() -> dict[str, set[str]]:
    import torch
    import torch.nn.functional as functional

    return {
        "torch": public_nonclass_callables(torch),
        "torch.Tensor": public_nonclass_callables(torch.Tensor),
        "torch.nn.functional": public_nonclass_callables(functional),
    }


def load_public_api() -> dict[str, set[str]]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:
        raise SystemExit(
            "This script needs PyTorch installed so it can inspect the public "
            "torch, torch.Tensor, torch.nn, and torch.nn.functional APIs."
        ) from exc

    return {
        "torch": public_callables(torch),
        "torch.Tensor": public_callables(torch.Tensor),
        "torch.nn": public_callables(nn),
        "torch.nn.functional": public_callables(functional),
    }


def resolve_transformers_package(path_arg: str | None) -> Path:
    if path_arg:
        root = Path(path_arg).expanduser().resolve()
        candidates = (
            root / "src" / "transformers",
            root / "transformers",
            root,
        )
        for candidate in candidates:
            if (candidate / "models").is_dir():
                return candidate
        raise SystemExit(f"Could not find a transformers/models directory under {root}")

    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise SystemExit(
            "No Transformers source directory was provided and the transformers "
            "package is not importable from site-packages."
        ) from exc

    package_dir = Path(transformers.__file__).resolve().parent
    if not (package_dir / "models").is_dir():
        raise SystemExit(f"Imported transformers, but no models directory exists at {package_dir}")
    return package_dir


def model_files(transformers_package: Path) -> list[Path]:
    models_dir = transformers_package / "models"
    excluded_prefixes = (
        "modeling_flax_",
        "modeling_jax_",
        "modeling_tf_",
    )
    return sorted(
        path
        for path in models_dir.glob("*/modeling_*.py")
        if not path.name.startswith(excluded_prefixes)
    )


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        if base:
            return f"{base}.{node.attr}"
    return None


def root_name(node: ast.AST) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    if isinstance(node, ast.Name):
        return node.id
    return None


def is_probable_tensor_method(node: ast.Attribute) -> bool:
    base_root = root_name(node.value)
    if base_root in IGNORED_TENSOR_BASES:
        return False
    if isinstance(node.value, ast.Name) and node.value.id[:1].isupper():
        return False
    return True


class TorchApiVisitor(ast.NodeVisitor):
    def __init__(self, public_api: dict[str, set[str]]) -> None:
        self.public_api = public_api
        self.aliases: dict[str, str] = {}
        self.used: dict[str, set[str]] = {key: set() for key in API_KEYS}

    def visit_Import(self, node: ast.Import) -> None:
        for item in node.names:
            if item.name == "torch":
                self.aliases[item.asname or "torch"] = "torch"
            elif item.name == "torch.nn":
                self.aliases[item.asname or "nn"] = "torch.nn"
            elif item.name == "torch.nn.functional":
                self.aliases[item.asname or "functional"] = "torch.nn.functional"
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if module == "torch":
            for item in node.names:
                if item.name == "*":
                    continue
                local = item.asname or item.name
                if item.name == "nn":
                    self.aliases[local] = "torch.nn"
                elif item.name in self.public_api["torch"]:
                    self.aliases[local] = f"torch.{item.name}"
        elif module == "torch.nn":
            for item in node.names:
                if item.name == "*":
                    continue
                local = item.asname or item.name
                if item.name == "functional":
                    self.aliases[local] = "torch.nn.functional"
                elif item.name in self.public_api["torch.nn"]:
                    self.aliases[local] = f"torch.nn.{item.name}"
        elif module == "torch.nn.functional":
            for item in node.names:
                if item.name == "*":
                    continue
                if item.name in self.public_api["torch.nn.functional"]:
                    self.aliases[item.asname or item.name] = f"torch.nn.functional.{item.name}"
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self._record_call(node.func)
        self.generic_visit(node)

    def _record_call(self, func: ast.AST) -> None:
        if isinstance(func, ast.Name):
            target = self.aliases.get(func.id)
            if target:
                self._record_target(target)
            return

        if not isinstance(func, ast.Attribute):
            return

        base = dotted_name(func.value)
        if base is not None:
            target_base = self.aliases.get(base, base)
            target = f"{target_base}.{func.attr}"
            if self._record_target(target):
                return

        if func.attr in self.public_api["torch.Tensor"] and is_probable_tensor_method(func):
            self.used["torch.Tensor"].add(func.attr)

    def _record_target(self, target: str) -> bool:
        for key in ("torch.nn.functional", "torch.nn", "torch"):
            prefix = f"{key}."
            if not target.startswith(prefix):
                continue
            name = target.removeprefix(prefix).split(".", 1)[0]
            if name in self.public_api[key]:
                self.used[key].add(name)
                return True
        return False


def scan_file(path: Path, public_api: dict[str, set[str]]) -> dict[str, set[str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        print(f"Skipping {path}: {exc}", file=sys.stderr)
        return {key: set() for key in API_KEYS}

    visitor = TorchApiVisitor(public_api)
    visitor.visit(tree)
    return visitor.used


def scan(
    transformers_package: Path,
    public_api: dict[str, set[str]] | None = None,
) -> dict[str, dict[str, Any]]:
    if public_api is None:
        public_api = load_public_api()
    families: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "files": [],
            **{key: set() for key in API_KEYS},
        }
    )

    for path in model_files(transformers_package):
        family = path.parent.name
        entry = families[family]
        entry["files"].append(str(path.relative_to(transformers_package)))
        file_used = scan_file(path, public_api)
        for key, values in file_used.items():
            entry[key].update(values)

    return {
        family: {
            "files": sorted(entry["files"]),
            **{key: sorted(entry[key]) for key in API_KEYS},
        }
        for family, entry in sorted(families.items())
    }


def write_json(data: dict[str, dict[str, Any]], output: Path | None) -> None:
    text = json.dumps(data, indent=2, sort_keys=True)
    if output:
        output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def rows(data: dict[str, dict[str, Any]]) -> Iterable[dict[str, str]]:
    for family, entry in data.items():
        for key in API_KEYS:
            for name in entry[key]:
                yield {"model_family": family, "api_group": key, "name": name}


def write_csv(data: dict[str, dict[str, Any]], output: Path | None) -> None:
    if output:
        handle = output.open("w", newline="", encoding="utf-8")
        close = True
    else:
        handle = sys.stdout
        close = False

    try:
        writer = csv.DictWriter(handle, fieldnames=("model_family", "api_group", "name"))
        writer.writeheader()
        writer.writerows(rows(data))
    finally:
        if close:
            handle.close()


def write_markdown(data: dict[str, dict[str, Any]], output: Path | None) -> None:
    lines: list[str] = []
    for family, entry in data.items():
        lines.append(f"## {family}")
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
    counts: dict[str, int] = defaultdict(int)
    for entry in data.values():
        names = set()
        for key in keys:
            for name in entry[key]:
                if normalize_inplace and name.endswith("_"):
                    name = name[:-1]
                names.add(name)
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
    lines = [
        f"## {label}",
        "",
    ]
    lines.extend(f"- `{name}`" for name in sorted(counts))
    lines.append("")
    return lines


def normalize_name(name: str, *, normalize_inplace: bool) -> str:
    if normalize_inplace and name.endswith("_"):
        return name[:-1]
    return name


def public_names(public_api: dict[str, set[str]], keys: tuple[str, ...], *, normalize_inplace: bool) -> set[str]:
    names: set[str] = set()
    for key in keys:
        for name in public_api[key]:
            names.add(normalize_name(name, normalize_inplace=normalize_inplace))
    return names


def write_aggregate_markdown(data: dict[str, dict[str, Any]], output: Path | None) -> None:
    lines = ["# Torch API Usage by Transformers Model Family", ""]
    torch_tensor_counts = aggregate_counts(data, ("torch", "torch.Tensor"), normalize_inplace=True)
    nn_counts = aggregate_counts(data, ("torch.nn",))
    functional_counts = aggregate_counts(data, ("torch.nn.functional",), normalize_inplace=True)

    lines.extend(
        markdown_count_table(
            "torch and torch.Tensor",
            "function",
            torch_tensor_counts,
        )
    )
    lines.extend(
        markdown_count_table(
            "torch.nn",
            "module",
            nn_counts,
        )
    )
    lines.extend(
        markdown_count_table(
            "torch.nn.functional",
            "function",
            functional_counts,
        )
    )
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
    public_function_api = load_public_function_api()
    sections = (
        (
            "Unused torch and torch.Tensor Functions",
            public_names(public_function_api, ("torch", "torch.Tensor"), normalize_inplace=True),
            set(aggregate_counts(data, ("torch", "torch.Tensor"), normalize_inplace=True)),
        ),
        (
            "Unused torch.nn Modules",
            public_nn_modules(),
            set(aggregate_counts(data, ("torch.nn",))),
        ),
        (
            "Unused torch.nn.functional Functions",
            public_names(public_function_api, ("torch.nn.functional",), normalize_inplace=True),
            set(aggregate_counts(data, ("torch.nn.functional",), normalize_inplace=True)),
        ),
    )

    lines = ["# Public Torch APIs Not Used by Transformers Model Files", ""]
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
            "used by each Transformers model family."
        )
    )
    parser.add_argument(
        "transformers_dir",
        nargs="?",
        help=(
            "Path to a Transformers repo root, src/transformers, or transformers "
            "package directory. Defaults to site-packages transformers."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("json", "csv", "markdown", "aggregate-markdown", "unused-markdown"),
        default="json",
        help="Output format. Default: json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write output to this file instead of stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transformers_package = resolve_transformers_package(args.transformers_dir)
    public_api = load_public_api()
    data = scan(transformers_package, public_api)

    if args.format == "json":
        write_json(data, args.output)
    elif args.format == "csv":
        write_csv(data, args.output)
    elif args.format == "markdown":
        write_markdown(data, args.output)
    elif args.format == "unused-markdown":
        write_unused_markdown(data, public_api, args.output)
    else:
        write_aggregate_markdown(data, args.output)


if __name__ == "__main__":
    main()
