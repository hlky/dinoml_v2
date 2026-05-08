from __future__ import annotations

from dataclasses import dataclass

from dinoml.backends.registry import get_backend_spec


@dataclass(frozen=True)
class Target:
    name: str
    arch: str | None = None

    def __post_init__(self) -> None:
        spec = get_backend_spec(self.name)
        arch = self.arch
        if arch is None or (self.name == "cpu" and arch == "sm_86"):
            arch = spec.default_arch
        object.__setattr__(self, "arch", arch)

    def to_json(self) -> dict[str, str]:
        return {"name": self.name, "arch": str(self.arch)}
