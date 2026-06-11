from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import dinoml as dml


ATOL_BY_DTYPE = {"float16": 1e-3, "float32": 1e-6}
RTOL_BY_DTYPE = {"float16": 1e-3, "float32": 1e-6}


@dataclass(frozen=True)
class TensorFilterHelperCase:
    name: str
    op_name: str
    dtype: str
    input_shape: tuple[int, ...] | None = None
    up: int = 2
    pad0: int = 2
    pad1: int = 1
    channels: int | None = None


TENSOR_FILTER_HELPER_CASES = (
    TensorFilterHelperCase("fir_downsample2d_f32", "fir_downsample2d", "float32", input_shape=(2, 7, 9, 5)),
    TensorFilterHelperCase("fir_downsample2d_f16", "fir_downsample2d", "float16", input_shape=(2, 8, 10, 4)),
    TensorFilterHelperCase("fir_filter_pad2_f32", "fir_filter_pad2", "float32", input_shape=(2, 5, 7, 3)),
    TensorFilterHelperCase("fir_filter_pad2_f16", "fir_filter_pad2", "float16", input_shape=(2, 6, 4, 6)),
    TensorFilterHelperCase("fir_upsample2d_default_f32", "fir_upsample2d", "float32", input_shape=(2, 5, 7, 4)),
    TensorFilterHelperCase("fir_upsample2d_default_f16", "fir_upsample2d", "float16", input_shape=(1, 6, 5, 3)),
    TensorFilterHelperCase(
        "fir_upsample2d_conv_path_f32",
        "fir_upsample2d",
        "float32",
        input_shape=(2, 9, 11, 5),
        up=1,
        pad0=1,
        pad1=1,
    ),
    TensorFilterHelperCase(
        "fir_upsample2d_conv_path_f16",
        "fir_upsample2d",
        "float16",
        input_shape=(1, 10, 8, 4),
        up=1,
        pad0=1,
        pad1=1,
    ),
    TensorFilterHelperCase("kdownsample2d_weight_f32", "kdownsample2d_weight", "float32", channels=5),
    TensorFilterHelperCase("kdownsample2d_weight_f16", "kdownsample2d_weight", "float16", channels=7),
    TensorFilterHelperCase("kupsample2d_weight_f32", "kupsample2d_weight", "float32", channels=6),
    TensorFilterHelperCase("kupsample2d_weight_f16", "kupsample2d_weight", "float16", channels=4),
)


class _TensorFilterHelperModule(dml.Module):
    def __init__(self, case: TensorFilterHelperCase):
        self.case = case

    def forward(self, x=None):
        if self.case.op_name == "fir_downsample2d":
            y = dml.ops.fir_downsample2d(x)
        elif self.case.op_name == "fir_filter_pad2":
            y = dml.ops.fir_filter_pad2(x)
        elif self.case.op_name == "fir_upsample2d":
            y = dml.ops.fir_upsample2d(x, up=self.case.up, pad0=self.case.pad0, pad1=self.case.pad1)
        elif self.case.op_name == "kdownsample2d_weight":
            y = dml.ops.kdownsample2d_weight(int(self.case.channels), dtype=self.case.dtype)
        elif self.case.op_name == "kupsample2d_weight":
            y = dml.ops.kupsample2d_weight(int(self.case.channels), dtype=self.case.dtype)
        else:
            raise ValueError(f"Unsupported tensor filter helper op {self.case.op_name!r}")
        return dml.ops.output(y, "y")


def trace_tensor_filter_helper_spec(case: TensorFilterHelperCase):
    inputs = {} if case.input_shape is None else {"x": dml.TensorSpec(list(case.input_shape), case.dtype)}
    return dml.trace(_TensorFilterHelperModule(case), inputs=inputs, name=f"{case.name}_parity")


def random_inputs(case: TensorFilterHelperCase, *, seed: int = 7) -> dict[str, np.ndarray]:
    if case.input_shape is None:
        return {}
    rng = np.random.default_rng(seed)
    value = rng.standard_normal(case.input_shape, dtype=np.float32).astype(np.float32, copy=False)
    if case.dtype == "float16":
        value = value.astype(np.float16)
    return {"x": value}


def numpy_oracle(case: TensorFilterHelperCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    if case.op_name == "fir_downsample2d":
        output = _fir_downsample2d_oracle(inputs["x"])
    elif case.op_name == "fir_filter_pad2":
        output = _fir_filter_pad2_oracle(inputs["x"])
    elif case.op_name == "fir_upsample2d":
        output = _fir_upsample2d_oracle(inputs["x"], up=case.up, pad0=case.pad0, pad1=case.pad1)
    elif case.op_name == "kdownsample2d_weight":
        output = _tensor_filter_weight_oracle(int(case.channels), (0.125, 0.375, 0.375, 0.125))
    elif case.op_name == "kupsample2d_weight":
        output = _tensor_filter_weight_oracle(int(case.channels), (0.25, 0.75, 0.75, 0.25))
    else:
        raise ValueError(f"Unsupported tensor filter helper op {case.op_name!r}")
    return _quantize_expected(output, case.dtype)


def _fir_downsample2d_oracle(x: np.ndarray) -> np.ndarray:
    source = np.asarray(x, dtype=np.float32)
    batch, in_h, in_w, channels = source.shape
    out_h = in_h // 2
    out_w = in_w // 2
    kernel = np.asarray((0.125, 0.375, 0.375, 0.125), dtype=np.float32)
    output = np.empty((batch, out_h, out_w, channels), dtype=np.float32)
    for n in range(batch):
        for out_y in range(out_h):
            for out_x in range(out_w):
                acc = np.zeros((channels,), dtype=np.float32)
                for kh in range(4):
                    in_y = out_y * 2 + kh - 1
                    if in_y < 0 or in_y >= in_h:
                        continue
                    for kw in range(4):
                        in_x = out_x * 2 + kw - 1
                        if in_x < 0 or in_x >= in_w:
                            continue
                        acc += source[n, in_y, in_x, :] * np.float32(kernel[kh] * kernel[kw])
                output[n, out_y, out_x, :] = acc
    return output


def _fir_filter_pad2_oracle(x: np.ndarray) -> np.ndarray:
    source = np.asarray(x, dtype=np.float32)
    batch, in_h, in_w, channels = source.shape
    out_h = in_h + 1
    out_w = in_w + 1
    kernel = np.asarray((0.125, 0.375, 0.375, 0.125), dtype=np.float32)
    output = np.empty((batch, out_h, out_w, channels), dtype=np.float32)
    for n in range(batch):
        for out_y in range(out_h):
            for out_x in range(out_w):
                acc = np.zeros((channels,), dtype=np.float32)
                for kh in range(4):
                    in_y = out_y + kh - 2
                    if in_y < 0 or in_y >= in_h:
                        continue
                    for kw in range(4):
                        in_x = out_x + kw - 2
                        if in_x < 0 or in_x >= in_w:
                            continue
                        acc += source[n, in_y, in_x, :] * np.float32(kernel[kh] * kernel[kw])
                output[n, out_y, out_x, :] = acc
    return output


def _fir_upsample2d_oracle(x: np.ndarray, *, up: int, pad0: int, pad1: int) -> np.ndarray:
    source = np.asarray(x, dtype=np.float32)
    batch, in_h, in_w, channels = source.shape
    out_h = in_h * int(up) + int(pad0) + int(pad1) - 3
    out_w = in_w * int(up) + int(pad0) + int(pad1) - 3
    kernel = np.asarray((0.25, 0.75, 0.75, 0.25), dtype=np.float32)
    output = np.empty((batch, out_h, out_w, channels), dtype=np.float32)
    for n in range(batch):
        for out_y in range(out_h):
            for out_x in range(out_w):
                acc = np.zeros((channels,), dtype=np.float32)
                for kh in range(4):
                    upsampled_y = out_y + kh - int(pad0)
                    if (upsampled_y % int(up)) != 0:
                        continue
                    in_y = upsampled_y // int(up)
                    if in_y < 0 or in_y >= in_h:
                        continue
                    for kw in range(4):
                        upsampled_x = out_x + kw - int(pad0)
                        if (upsampled_x % int(up)) != 0:
                            continue
                        in_x = upsampled_x // int(up)
                        if in_x < 0 or in_x >= in_w:
                            continue
                        acc += source[n, in_y, in_x, :] * np.float32(kernel[kh] * kernel[kw])
                output[n, out_y, out_x, :] = acc
    return output


def _tensor_filter_weight_oracle(channels: int, kernel_1d: tuple[float, float, float, float]) -> np.ndarray:
    output = np.zeros((channels, channels, 4, 4), dtype=np.float32)
    kernel = np.outer(np.asarray(kernel_1d, dtype=np.float32), np.asarray(kernel_1d, dtype=np.float32))
    for channel in range(channels):
        output[channel, channel, :, :] = kernel
    return output


def _quantize_expected(value: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "float16":
        return np.asarray(value, dtype=np.float16).astype(np.float32)
    if dtype == "float32":
        return np.asarray(value, dtype=np.float32)
    raise ValueError(f"Unsupported tensor filter helper dtype {dtype!r}")
