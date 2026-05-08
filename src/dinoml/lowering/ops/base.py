from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


RenderKernelFn = Callable[[str, Mapping[str, Any], Mapping[str, Mapping[str, Any]]], str | None]
RenderLaunchFn = Callable[[str, Mapping[str, Any], Mapping[str, Mapping[str, Any]]], str]
SourceKeyFn = Callable[[str, Mapping[str, Any], Mapping[str, Mapping[str, Any]]], str]


@dataclass(frozen=True)
class OpLowering:
    op_name: str
    render_generated_kernel: RenderKernelFn
    render_launch: RenderLaunchFn
    source_key: SourceKeyFn | None = None
