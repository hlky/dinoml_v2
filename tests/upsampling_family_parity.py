from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import dinoml as dml


ATOL_BY_DTYPE = {"float16": 0.01, "float32": 1e-5, "bfloat16": 0.03}
RTOL_BY_DTYPE = {"float16": 0.01, "float32": 1e-5, "bfloat16": 0.03}


@dataclass(frozen=True)
class UpsamplingParityCase:
    name: str
    op_name: str
    input_shape: tuple[int, ...]
    scale_factor: float | None = None
    mode: str | None = None
    align_corners: bool | None = None
    residual_shape: tuple[int, ...] | None = None


UPSAMPLING_PARITY_CASES = (
    UpsamplingParityCase("upsampling1d_linear", "upsampling1d", (2, 8, 6), 3.5, "linear", False),
    UpsamplingParityCase("upsampling1d_linear_align_corners", "upsampling1d", (2, 8, 6), 2.0, "linear", True),
    UpsamplingParityCase("upsampling1d_nearest", "upsampling1d", (2, 8, 7), 2.0, "nearest", None),
    UpsamplingParityCase("upsampling1d_nearest_exact", "upsampling1d", (2, 8, 7), 2.0, "nearest-exact", None),
    UpsamplingParityCase("upsampling1d_add_linear", "upsampling1d_add", (2, 8, 6), 2.0, "linear", False, (2, 16, 6)),
    UpsamplingParityCase("upsampling1d_add_nearest_exact", "upsampling1d_add", (2, 8, 7), 2.0, "nearest-exact", None, (2, 16, 7)),
    UpsamplingParityCase("upsampling2d_bilinear", "upsampling2d", (2, 5, 7, 6), 3.5, "bilinear", False),
    UpsamplingParityCase("upsampling2d_bilinear_align_corners", "upsampling2d", (2, 5, 7, 6), 2.0, "bilinear", True),
    UpsamplingParityCase("upsampling2d_nearest", "upsampling2d", (2, 5, 7, 7), 2.0, "nearest", None),
    UpsamplingParityCase("upsampling2d_nearest_exact", "upsampling2d", (2, 5, 7, 7), 2.0, "nearest-exact", None),
    UpsamplingParityCase("upsampling2d_add_bilinear", "upsampling2d_add", (2, 5, 7, 6), 2.0, "bilinear", False, (2, 10, 14, 6)),
    UpsamplingParityCase("upsampling2d_add_nearest", "upsampling2d_add", (2, 5, 7, 7), 2.0, "nearest", None, (2, 10, 14, 7)),
    UpsamplingParityCase("upsampling3d_trilinear", "upsampling3d", (1, 4, 5, 7, 6), 2.0, "trilinear", False),
    UpsamplingParityCase("upsampling3d_trilinear_align_corners", "upsampling3d", (1, 4, 5, 7, 6), 2.0, "trilinear", True),
    UpsamplingParityCase("upsampling3d_nearest", "upsampling3d", (1, 4, 5, 7, 7), 2.0, "nearest", None),
    UpsamplingParityCase("upsampling3d_nearest_exact", "upsampling3d", (1, 4, 5, 7, 7), 2.0, "nearest-exact", None),
    UpsamplingParityCase("upsampling3d_add_trilinear", "upsampling3d_add", (1, 4, 5, 7, 6), 2.0, "trilinear", False, (1, 8, 10, 14, 6)),
    UpsamplingParityCase("upsampling3d_add_nearest_exact", "upsampling3d_add", (1, 4, 5, 7, 7), 2.0, "nearest-exact", None, (1, 8, 10, 14, 7)),
)

COMPRESS_TIME_FRAME_SIZES = (1, 3, 8)


class _UpsamplingParityModule(dml.Module):
    def __init__(self, case: UpsamplingParityCase):
        self.case = case

    def forward(self, x, residual=None):
        kwargs = {}
        if self.case.align_corners is not None:
            kwargs["align_corners"] = self.case.align_corners
        elif self.case.op_name != "upsampling3d_compress_time":
            kwargs["align_corners"] = None
        if self.case.op_name == "upsampling1d":
            y = dml.ops.upsampling1d(x, self.case.scale_factor, self.case.mode, **kwargs)
        elif self.case.op_name == "upsampling1d_add":
            y = dml.ops.upsampling1d_add(x, residual, self.case.scale_factor, self.case.mode, **kwargs)
        elif self.case.op_name == "upsampling2d":
            y = dml.ops.upsampling2d(x, self.case.scale_factor, self.case.mode, **kwargs)
        elif self.case.op_name == "upsampling2d_add":
            y = dml.ops.upsampling2d_add(x, residual, self.case.scale_factor, self.case.mode, **kwargs)
        elif self.case.op_name == "upsampling3d":
            y = dml.ops.upsampling3d(x, self.case.scale_factor, self.case.mode, **kwargs)
        elif self.case.op_name == "upsampling3d_add":
            y = dml.ops.upsampling3d_add(x, residual, self.case.scale_factor, self.case.mode, **kwargs)
        elif self.case.op_name == "upsampling3d_compress_time":
            y = dml.ops.upsampling3d_compress_time(x)
        else:
            raise ValueError(f"Unsupported upsampling parity op {self.case.op_name!r}")
        return dml.ops.output(y, "y")


def trace_upsampling_parity_spec(case: UpsamplingParityCase, dtype: str):
    inputs = {"x": dml.TensorSpec(list(case.input_shape), dtype)}
    if case.residual_shape is not None:
        inputs["residual"] = dml.TensorSpec(list(case.residual_shape), dtype)
    return dml.trace(_UpsamplingParityModule(case), inputs=inputs, name=f"{case.name}_{dtype}_parity")


def random_inputs(case: UpsamplingParityCase, dtype: str, *, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = {"x": rng.standard_normal(case.input_shape, dtype=np.float32).astype(np.float32)}
    if case.residual_shape is not None:
        values["residual"] = rng.standard_normal(case.residual_shape, dtype=np.float32).astype(np.float32)
    if dtype == "float16":
        return {name: value.astype(np.float16) for name, value in values.items()}
    return values


def torch_oracle(torch, case: UpsamplingParityCase, inputs: dict[str, np.ndarray], *, device, dtype: str, native_dtype: bool) -> np.ndarray:
    torch_dtype = _torch_dtype(torch, dtype, native_dtype=native_dtype)
    x = _to_torch_input(torch, inputs["x"], device=device, dtype=torch_dtype)
    if case.op_name == "upsampling3d_compress_time":
        result = _compress_time_oracle(torch, x, device=device, dtype=torch_dtype)
    elif case.op_name.startswith("upsampling1d"):
        result = torch.nn.functional.interpolate(
            x.permute(0, 2, 1),
            scale_factor=case.scale_factor,
            mode=case.mode,
            align_corners=case.align_corners,
        ).permute(0, 2, 1)
    elif case.op_name.startswith("upsampling2d"):
        result = torch.nn.functional.interpolate(
            x.permute(0, 3, 1, 2),
            scale_factor=case.scale_factor,
            mode=case.mode,
            align_corners=case.align_corners,
        ).permute(0, 2, 3, 1)
    elif case.op_name.startswith("upsampling3d"):
        result = torch.nn.functional.interpolate(
            x.permute(0, 4, 1, 2, 3),
            scale_factor=case.scale_factor,
            mode=case.mode,
            align_corners=case.align_corners,
        ).permute(0, 2, 3, 4, 1)
    else:
        raise ValueError(f"Unsupported upsampling parity op {case.op_name!r}")
    if case.residual_shape is not None:
        residual = _to_torch_input(torch, inputs["residual"], device=device, dtype=torch_dtype)
        result = result + residual
    return result.float().cpu().numpy()


def _compress_time_oracle(torch, x, *, device, dtype):
    y = x.permute(0, 4, 1, 2, 3)
    frames = int(y.shape[2])
    if frames > 1 and (frames % 2) == 1:
        first = torch.nn.functional.interpolate(y[:, :, 0], scale_factor=2.0, mode="nearest")
        rest = torch.nn.functional.interpolate(y[:, :, 1:], scale_factor=2.0, mode="nearest")
        return torch.cat([first[:, :, None, :, :], rest], dim=2).permute(0, 2, 3, 4, 1)
    if frames > 1:
        return torch.nn.functional.interpolate(y, scale_factor=2.0, mode="nearest").permute(0, 2, 3, 4, 1)
    spatial = torch.nn.functional.interpolate(y.squeeze(2), scale_factor=2.0, mode="nearest")
    return spatial[:, :, None, :, :].permute(0, 2, 3, 4, 1)


def _to_torch_input(torch, value: np.ndarray, *, device, dtype):
    return torch.from_numpy(np.asarray(value)).to(device=device, dtype=dtype)


def _torch_dtype(torch, dtype: str, *, native_dtype: bool):
    if not native_dtype:
        return torch.float32
    return {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
