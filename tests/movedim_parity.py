from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import dinoml as dml


@dataclass(frozen=True)
class MovedimCase:
    name: str
    shape: tuple[int, ...]
    source: int | tuple[int, ...]
    destination: int | tuple[int, ...]
    use_method: bool = False


MOVEDIM_CASES = (
    MovedimCase(
        name="movedim_single_int_method",
        shape=(2, 3, 4),
        source=0,
        destination=-1,
        use_method=True,
    ),
    MovedimCase(
        name="movedim_sequence_function",
        shape=(2, 3, 4, 5),
        source=(0, 3),
        destination=(2, 0),
    ),
    MovedimCase(
        name="movedim_negative_sequence",
        shape=(2, 3, 4, 5),
        source=(-1, 1),
        destination=(0, -1),
    ),
)


class _MovedimModule(dml.Module):
    def __init__(self, case: MovedimCase):
        self.case = case

    def forward(self, x):
        if self.case.use_method:
            y = x.movedim(self.case.source, self.case.destination)
        else:
            y = dml.ops.movedim(x, self.case.source, self.case.destination)
        return dml.ops.output(y, "y")


def trace_movedim_spec(case: MovedimCase):
    return dml.trace(
        _MovedimModule(case),
        inputs={"x": dml.TensorSpec(list(case.shape), "float32")},
        name=f"{case.name}_parity",
    )


def random_inputs(case: MovedimCase, *, seed: int = 17) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(case.shape, dtype=np.float32).astype(np.float32, copy=False)
    return {"x": x}


def torch_oracle(case: MovedimCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    torch = __import__("torch")
    x = torch.tensor(inputs["x"], dtype=torch.float32)
    return torch.movedim(x, case.source, case.destination).cpu().numpy().astype(np.float32, copy=False)
