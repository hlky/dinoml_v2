from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    name: str
    arch: str = "sm_86"

    def __post_init__(self) -> None:
        if self.name not in {"cuda", "cpu"}:
            raise ValueError("DinoML v2 currently supports target='cuda' or target='cpu'")
        if self.name == "cpu" and self.arch == "sm_86":
            object.__setattr__(self, "arch", "native")

    def to_json(self) -> dict[str, str]:
        return {"name": self.name, "arch": self.arch}
