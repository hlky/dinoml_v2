from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dinoml.backends.registry import get_backend_spec


@dataclass(frozen=True)
class Target:
    name: str
    arch: str | None = None
    no_tf32: bool = False
    use_fp16_acc: bool = False

    def __post_init__(self) -> None:
        spec = get_backend_spec(self.name)
        arch = self.arch
        if arch is None or (self.name == "cpu" and arch == "sm_86"):
            arch = spec.default_arch
        object.__setattr__(self, "arch", arch)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arch": str(self.arch),
            "no_tf32": bool(self.no_tf32),
            "use_fp16_acc": bool(self.use_fp16_acc),
        }
