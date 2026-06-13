from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from dinoml import ops
from dinoml.frontend import Module as _FrontendModule
from dinoml.frontend import Parameter, Tensor, as_tensor
from dinoml.ir import normalize_dtype
from . import functional


class Module(_FrontendModule):
    """Lightweight torch.nn-style module base for traced DinoML models."""

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter) and value.name is None:
            _set_auto_parameter_name(value, name)
        super().__setattr__(name, value)
        if isinstance(value, _FrontendModule) and not name.startswith("_"):
            _prefix_auto_parameter_names(value, name)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for _, parameter in self.named_parameters(recurse=recurse):
            yield parameter

    def named_parameters(self, prefix: str = "", recurse: bool = True) -> Iterator[tuple[str, Parameter]]:
        seen: set[int] = set()
        yield from _iter_named_parameters(self, prefix, recurse, seen)

    def children(self) -> Iterator[_FrontendModule]:
        for _, child in self.named_children():
            yield child

    def named_children(self) -> Iterator[tuple[str, _FrontendModule]]:
        for name, value in vars(self).items():
            if not name.startswith("_") and isinstance(value, _FrontendModule):
                yield name, value


class Identity(Module):
    def forward(self, x: Any) -> Any:
        return x


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: str = "float32",
        specialization: str | None = None,
    ):
        self.in_features = _positive_int(in_features, "Linear in_features")
        self.out_features = _positive_int(out_features, "Linear out_features")
        self.dtype = normalize_dtype(dtype)
        self.specialization = specialization
        self.weight = Parameter([self.out_features, self.in_features], dtype=self.dtype)
        self.bias = Parameter([self.out_features], dtype=self.dtype) if bias else None
        self.in_channels = self.in_features
        self.out_channels = self.out_features
        if specialization is not None and not isinstance(specialization, str):
            raise TypeError("Linear specialization must be a string or None")
        if specialization and not bias:
            raise NotImplementedError("Linear specialization currently requires bias=True")

    def forward(self, x: Any, *epilogue_inputs: Any) -> Tensor:
        if self.bias is None:
            if epilogue_inputs:
                raise ValueError("Linear without bias does not accept epilogue inputs")
            return ops.gemm_rcr(x, self.weight)
        op_name = "gemm_rcr_bias" if self.specialization is None else f"gemm_rcr_bias_{self.specialization}"
        try:
            op = getattr(ops, op_name)
        except AttributeError as exc:
            raise ValueError(f"Unsupported Linear specialization {self.specialization!r}") from exc
        return op(x, self.weight, self.bias, *epilogue_inputs)


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: str = "float32",
        activation: str | None = None,
        add: bool = False,
    ):
        self.in_channels = _positive_int(in_channels, "Conv2d in_channels")
        self.out_channels = _positive_int(out_channels, "Conv2d out_channels")
        self.kernel_size = _pair(kernel_size, "Conv2d kernel_size", positive=True)
        self.stride = _pair(stride, "Conv2d stride", positive=True)
        self.padding = _pair(padding, "Conv2d padding", non_negative=True)
        self.dilation = _pair(dilation, "Conv2d dilation", positive=True)
        self.groups = _positive_int(groups, "Conv2d groups")
        if self.in_channels % self.groups != 0:
            raise ValueError("Conv2d in_channels must be divisible by groups")
        if activation not in {None, "relu"}:
            raise NotImplementedError("Conv2d currently supports activation=None or activation='relu'")
        self.activation = activation
        self.add = bool(add)
        self.dtype = normalize_dtype(dtype)
        self.weight = Parameter(
            [self.out_channels, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]],
            dtype=self.dtype,
        )
        self.bias = Parameter([self.out_channels], dtype=self.dtype) if bias else None

    def forward(self, x: Any, residual: Any | None = None) -> Tensor:
        if self.add and residual is None:
            raise ValueError("Conv2d residual is required when add=True")
        if residual is not None and not self.add:
            raise ValueError("Conv2d residual was provided but add=False")
        kwargs = {
            "stride": self.stride,
            "padding": self.padding,
            "dilation": self.dilation,
            "groups": self.groups,
        }
        if self.bias is not None:
            if self.add and self.activation == "relu":
                return ops.conv2d_bias_add_relu(x, self.weight, self.bias, residual, **kwargs)
            if self.add:
                return ops.conv2d_bias_add(x, self.weight, self.bias, residual, **kwargs)
            if self.activation == "relu":
                return ops.conv2d_bias_relu(x, self.weight, self.bias, **kwargs)
            return ops.conv2d_bias(x, self.weight, self.bias, **kwargs)
        y = ops.conv2d(x, self.weight, **kwargs)
        if self.add:
            y = ops.add(y, residual)
        if self.activation == "relu":
            y = ops.relu(y)
        return y


class Conv1d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: str = "float32",
    ):
        self.in_channels = _positive_int(in_channels, "Conv1d in_channels")
        self.out_channels = _positive_int(out_channels, "Conv1d out_channels")
        self.kernel_size = _single(kernel_size, "Conv1d kernel_size", positive=True)
        self.stride = _single(stride, "Conv1d stride", positive=True)
        self.padding = _single(padding, "Conv1d padding", non_negative=True)
        self.dilation = _single(dilation, "Conv1d dilation", positive=True)
        self.groups = _positive_int(groups, "Conv1d groups")
        if self.in_channels % self.groups != 0:
            raise ValueError("Conv1d in_channels must be divisible by groups")
        if not bias:
            raise NotImplementedError("Conv1d currently requires bias=True")
        self.dtype = normalize_dtype(dtype)
        self.weight = Parameter(
            [self.out_channels, self.in_channels // self.groups, self.kernel_size[0]],
            dtype=self.dtype,
        )
        self.bias = Parameter([self.out_channels], dtype=self.dtype)

    def forward(self, x: Any) -> Tensor:
        return ops.conv1d_bias(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class Conv3d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: str = "float32",
    ):
        self.in_channels = _positive_int(in_channels, "Conv3d in_channels")
        self.out_channels = _positive_int(out_channels, "Conv3d out_channels")
        self.kernel_size = _triple(kernel_size, "Conv3d kernel_size", positive=True)
        self.stride = _triple(stride, "Conv3d stride", positive=True)
        self.padding = _triple(padding, "Conv3d padding", non_negative=True)
        self.dilation = _triple(dilation, "Conv3d dilation", positive=True)
        self.groups = _positive_int(groups, "Conv3d groups")
        if self.in_channels % self.groups != 0:
            raise ValueError("Conv3d in_channels must be divisible by groups")
        self.dtype = normalize_dtype(dtype)
        self.weight = Parameter(
            [
                self.out_channels,
                self.in_channels // self.groups,
                self.kernel_size[0],
                self.kernel_size[1],
                self.kernel_size[2],
            ],
            dtype=self.dtype,
        )
        self.bias = Parameter([self.out_channels], dtype=self.dtype) if bias else None

    def forward(self, x: Any) -> Tensor:
        kwargs = {
            "stride": self.stride,
            "padding": self.padding,
            "dilation": self.dilation,
            "groups": self.groups,
        }
        if self.bias is None:
            return ops.conv3d(x, self.weight, **kwargs)
        return ops.conv3d_bias(x, self.weight, self.bias, **kwargs)


class ConvTranspose1d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        output_padding: int | Sequence[int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | Sequence[int] = 1,
        dtype: str = "float32",
    ):
        self.in_channels = _positive_int(in_channels, "ConvTranspose1d in_channels")
        self.out_channels = _positive_int(out_channels, "ConvTranspose1d out_channels")
        self.kernel_size = _single(kernel_size, "ConvTranspose1d kernel_size", positive=True)
        self.stride = _single(stride, "ConvTranspose1d stride", positive=True)
        self.padding = _single(padding, "ConvTranspose1d padding", non_negative=True)
        self.output_padding = _single(output_padding, "ConvTranspose1d output_padding", non_negative=True)
        self.groups = _positive_int(groups, "ConvTranspose1d groups")
        self.dilation = _single(dilation, "ConvTranspose1d dilation", positive=True)
        if self.in_channels % self.groups != 0:
            raise ValueError("ConvTranspose1d in_channels must be divisible by groups")
        if self.out_channels % self.groups != 0:
            raise ValueError("ConvTranspose1d out_channels must be divisible by groups")
        if bias:
            raise NotImplementedError("ConvTranspose1d currently requires bias=False")
        self.dtype = normalize_dtype(dtype)
        self.weight = Parameter(
            [self.in_channels, self.out_channels // self.groups, self.kernel_size[0]],
            dtype=self.dtype,
        )

    def forward(self, x: Any) -> Tensor:
        return ops.transposed_conv1d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class ConvTranspose2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        output_padding: int | Sequence[int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | Sequence[int] = 1,
        dtype: str = "float32",
    ):
        self.in_channels = _positive_int(in_channels, "ConvTranspose2d in_channels")
        self.out_channels = _positive_int(out_channels, "ConvTranspose2d out_channels")
        self.kernel_size = _pair(kernel_size, "ConvTranspose2d kernel_size", positive=True)
        self.stride = _pair(stride, "ConvTranspose2d stride", positive=True)
        self.padding = _pair(padding, "ConvTranspose2d padding", non_negative=True)
        self.output_padding = _pair(output_padding, "ConvTranspose2d output_padding", non_negative=True)
        self.groups = _positive_int(groups, "ConvTranspose2d groups")
        self.dilation = _pair(dilation, "ConvTranspose2d dilation", positive=True)
        if self.in_channels % self.groups != 0:
            raise ValueError("ConvTranspose2d in_channels must be divisible by groups")
        if self.out_channels % self.groups != 0:
            raise ValueError("ConvTranspose2d out_channels must be divisible by groups")
        if bias:
            raise NotImplementedError("ConvTranspose2d currently requires bias=False")
        self.dtype = normalize_dtype(dtype)
        self.weight = Parameter(
            [self.in_channels, self.out_channels // self.groups, self.kernel_size[0], self.kernel_size[1]],
            dtype=self.dtype,
        )

    def forward(self, x: Any) -> Tensor:
        return ops.transposed_conv2d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, dtype: str = "float32"):
        self.num_embeddings = _positive_int(num_embeddings, "Embedding num_embeddings")
        self.embedding_dim = _positive_int(embedding_dim, "Embedding embedding_dim")
        self.dtype = normalize_dtype(dtype)
        self.weight = Parameter([self.num_embeddings, self.embedding_dim], dtype=self.dtype)

    def forward(self, input: Any) -> Tensor:
        return ops.embedding(self.weight, input)


class GroupNorm(Module):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        dtype: str = "float32",
    ):
        self.num_groups = _positive_int(num_groups, "GroupNorm num_groups")
        self.num_channels = _positive_int(num_channels, "GroupNorm num_channels")
        if self.num_channels % self.num_groups != 0:
            raise ValueError("GroupNorm num_channels must be divisible by num_groups")
        self.eps = float(eps)
        self.affine = bool(affine)
        self.dtype = normalize_dtype(dtype)
        self.weight = Parameter([self.num_channels], dtype=self.dtype) if self.affine else None
        self.bias = Parameter([self.num_channels], dtype=self.dtype) if self.affine else None

    def forward(self, x: Any) -> Tensor:
        return ops.group_norm(x, self.num_groups, self.weight, self.bias, eps=self.eps)


class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        dtype: str = "float32",
    ):
        hidden = _single_normalized_shape(normalized_shape, "LayerNorm normalized_shape")
        if not elementwise_affine:
            raise NotImplementedError("LayerNorm currently requires elementwise_affine=True")
        self.normalized_shape = (hidden,)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        self.dtype = normalize_dtype(dtype)
        self.weight = Parameter([hidden], dtype=self.dtype)
        self.bias = Parameter([hidden], dtype=self.dtype)

    def forward(self, x: Any) -> Tensor:
        return ops.layer_norm(x, self.weight, self.bias, eps=self.eps)


class RMSNorm(Module):
    def __init__(self, normalized_shape: int | Sequence[int], eps: float = 1e-6, dtype: str = "float32"):
        hidden = _single_normalized_shape(normalized_shape, "RMSNorm normalized_shape")
        self.normalized_shape = (hidden,)
        self.eps = float(eps)
        self.dtype = normalize_dtype(dtype)
        self.weight = Parameter([hidden], dtype=self.dtype)

    def forward(self, x: Any) -> Tensor:
        return ops.rms_norm(x, self.weight, eps=self.eps)


class T5LayerNorm(RMSNorm):
    def forward(self, x: Any) -> Tensor:
        return ops.t5_layer_norm(x, self.weight, eps=self.eps)


class AvgPool1d(Module):
    def __init__(self, kernel_size: int | Sequence[int], stride: int | Sequence[int] | None = None, padding: int | Sequence[int] = 0):
        self.kernel_size = _single(kernel_size, "AvgPool1d kernel_size", positive=True)
        self.stride = None if stride is None else _single(stride, "AvgPool1d stride", positive=True)
        self.padding = _single(padding, "AvgPool1d padding", non_negative=True)

    def forward(self, x: Any) -> Tensor:
        return ops.avg_pool1d(x, self.kernel_size, stride=self.stride, padding=self.padding)


class AvgPool2d(Module):
    def __init__(self, kernel_size: int | Sequence[int], stride: int | Sequence[int] | None = None, padding: int | Sequence[int] = 0):
        self.kernel_size = _pair(kernel_size, "AvgPool2d kernel_size", positive=True)
        self.stride = None if stride is None else _pair(stride, "AvgPool2d stride", positive=True)
        self.padding = _pair(padding, "AvgPool2d padding", non_negative=True)

    def forward(self, x: Any) -> Tensor:
        return ops.avg_pool2d(x, self.kernel_size, stride=self.stride, padding=self.padding)


class MaxPool2d(Module):
    def __init__(self, kernel_size: int | Sequence[int], stride: int | Sequence[int] | None = None, padding: int | Sequence[int] = 0):
        self.kernel_size = _pair(kernel_size, "MaxPool2d kernel_size", positive=True)
        self.stride = None if stride is None else _pair(stride, "MaxPool2d stride", positive=True)
        self.padding = _pair(padding, "MaxPool2d padding", non_negative=True)

    def forward(self, x: Any) -> Tensor:
        return ops.max_pool2d(x, self.kernel_size, stride=self.stride, padding=self.padding)


class RoiAlign(Module):
    def __init__(
        self,
        pooled_size: int | Sequence[int],
        spatial_scale: float = 1.0,
        sampling_ratio: int = 0,
        position_sensitive: bool = False,
        continuous_coordinate: bool = False,
    ):
        self.pooled_size = _pair(pooled_size, "RoiAlign pooled_size", positive=True)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.position_sensitive = bool(position_sensitive)
        self.continuous_coordinate = bool(continuous_coordinate)

    def forward(self, x: Any, rois: Any) -> Tensor:
        return ops.roi_align(
            x,
            rois,
            pooled_size=self.pooled_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio,
            position_sensitive=self.position_sensitive,
            continuous_coordinate=self.continuous_coordinate,
        )


class MultiLevelRoiAlign(Module):
    def __init__(
        self,
        pooled_size: int | Sequence[int],
        im_shape: Sequence[int],
        spatial_scale: float = 1.0,
        sampling_ratio: int = 0,
        position_sensitive: bool = False,
        continuous_coordinate: bool = False,
    ):
        self.pooled_size = _pair(pooled_size, "MultiLevelRoiAlign pooled_size", positive=True)
        self.im_shape = _pair(im_shape, "MultiLevelRoiAlign im_shape", positive=True)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.position_sensitive = bool(position_sensitive)
        self.continuous_coordinate = bool(continuous_coordinate)

    def forward(self, p2: Any, p3: Any, p4: Any, p5: Any, rois: Any) -> Tensor:
        return ops.multi_level_roi_align(
            p2,
            p3,
            p4,
            p5,
            rois,
            pooled_size=self.pooled_size,
            im_shape=self.im_shape,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio,
            position_sensitive=self.position_sensitive,
            continuous_coordinate=self.continuous_coordinate,
        )


class Upsampling1d(Module):
    def __init__(self, scale_factor: float, mode: str, align_corners: bool | None = False):
        self.scale_factor = float(scale_factor)
        self.mode = str(mode)
        self.align_corners = align_corners

    def forward(self, x: Any) -> Tensor:
        return ops.upsampling1d(x, self.scale_factor, self.mode, align_corners=self.align_corners)


class Upsampling1dAdd(Module):
    def __init__(self, scale_factor: float, mode: str, align_corners: bool | None = False):
        self.scale_factor = float(scale_factor)
        self.mode = str(mode)
        self.align_corners = align_corners

    def forward(self, x: Any, residual: Any) -> Tensor:
        return ops.upsampling1d_add(x, residual, self.scale_factor, self.mode, align_corners=self.align_corners)


class Upsampling2d(Module):
    def __init__(self, scale_factor: float, mode: str, align_corners: bool | None = False):
        self.scale_factor = float(scale_factor)
        self.mode = str(mode)
        self.align_corners = align_corners

    def forward(self, x: Any) -> Tensor:
        return ops.upsampling2d(x, self.scale_factor, self.mode, align_corners=self.align_corners)


class Upsampling2dAdd(Module):
    def __init__(self, scale_factor: float, mode: str, align_corners: bool | None = False):
        self.scale_factor = float(scale_factor)
        self.mode = str(mode)
        self.align_corners = align_corners

    def forward(self, x: Any, residual: Any) -> Tensor:
        return ops.upsampling2d_add(x, residual, self.scale_factor, self.mode, align_corners=self.align_corners)


class Upsampling3d(Module):
    def __init__(self, scale_factor: float, mode: str, align_corners: bool | None = False):
        self.scale_factor = float(scale_factor)
        self.mode = str(mode)
        self.align_corners = align_corners

    def forward(self, x: Any) -> Tensor:
        return ops.upsampling3d(x, self.scale_factor, self.mode, align_corners=self.align_corners)


class Upsampling3dAdd(Module):
    def __init__(self, scale_factor: float, mode: str, align_corners: bool | None = False):
        self.scale_factor = float(scale_factor)
        self.mode = str(mode)
        self.align_corners = align_corners

    def forward(self, x: Any, residual: Any) -> Tensor:
        return ops.upsampling3d_add(x, residual, self.scale_factor, self.mode, align_corners=self.align_corners)


class Upsampling3dCompressTime(Module):
    def forward(self, x: Any) -> Tensor:
        return ops.upsampling3d_compress_time(x)


class Upsample(Module):
    def __init__(
        self,
        size: int | Sequence[int] | None = None,
        scale_factor: float | Sequence[float] | None = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
        recompute_scale_factor: bool | None = None,
    ):
        if size is None and scale_factor is None:
            raise ValueError("Upsample requires either size= or scale_factor=")
        if size is not None and scale_factor is not None:
            raise NotImplementedError("Upsample does not support specifying both size= and scale_factor=")
        if recompute_scale_factor not in (None, False):
            raise NotImplementedError("Upsample does not support recompute_scale_factor=True")
        self.size = size
        self.scale_factor = scale_factor
        self.mode = str(mode)
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, x: Any) -> Tensor:
        scale_factor = self.scale_factor
        if self.size is not None:
            scale_factor = _upsample_scale_factor_from_size(x, self.size)
        return ops.interpolate(
            x,
            size=None,
            scale_factor=scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
            antialias=False,
        )


class ReLU(Module):
    def __init__(self, inplace: bool = False):
        if inplace:
            raise NotImplementedError("ReLU(inplace=True) is not supported by the tracing frontend")
        self.inplace = False

    def forward(self, x: Any) -> Tensor:
        return ops.relu(x)


class GELU(Module):
    def __init__(self, approximate: str = "tanh"):
        if approximate not in {"none", "tanh"}:
            raise ValueError("GELU approximate must be 'none' or 'tanh'")
        self.approximate = approximate

    def forward(self, x: Any) -> Tensor:
        return ops.gelu(x, approximation=self.approximate)


class SiLU(Module):
    def forward(self, x: Any) -> Tensor:
        return ops.silu(x)


class Sigmoid(Module):
    def forward(self, x: Any) -> Tensor:
        return ops.sigmoid(x)


class Tanh(Module):
    def forward(self, x: Any) -> Tensor:
        return ops.tanh(x)


class Softmax(Module):
    def __init__(self, dim: int | None = None):
        self.dim = -1 if dim is None else int(dim)

    def forward(self, x: Any) -> Tensor:
        return ops.softmax(x, dim=self.dim)


class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        self.start_dim = int(start_dim)
        self.end_dim = int(end_dim)

    def forward(self, x: Any) -> Tensor:
        return ops.flatten(x, start_dim=self.start_dim, end_dim=self.end_dim)


class Unflatten(Module):
    def __init__(self, dim: int, unflattened_size: Sequence[Any]):
        self.dim = int(dim)
        self.unflattened_size = tuple(unflattened_size)

    def forward(self, x: Any) -> Tensor:
        return ops.unflatten(x, self.dim, self.unflattened_size)


class Dropout(Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        p_value = float(p)
        if p_value < 0.0 or p_value > 1.0:
            raise ValueError("Dropout p must be between 0 and 1")
        if inplace:
            raise NotImplementedError("Dropout(inplace=True) is not supported by the tracing frontend")
        self.p = p_value
        self.inplace = False

    def forward(self, x: Any) -> Any:
        return x


class ModuleList(Module, Sequence[_FrontendModule]):
    def __init__(self, modules: Iterable[_FrontendModule] = ()):
        self._modules = []
        for module in modules:
            self.append(module)

    def append(self, module: _FrontendModule) -> None:
        if not isinstance(module, _FrontendModule):
            raise TypeError(f"ModuleList items must be Module instances, got {type(module).__name__}")
        index = len(self._modules)
        self._modules.append(module)
        parent_prefix = getattr(self, "_dinoml_auto_prefix", None)
        parameter_prefix = f"{parent_prefix}_{index}" if parent_prefix else str(index)
        _prefix_auto_parameter_names(module, parameter_prefix)

    def __getitem__(self, index: int) -> _FrontendModule:
        return self._modules[index]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[_FrontendModule]:
        return iter(self._modules)


class Sequential(Module):
    def __init__(self, *modules: _FrontendModule):
        self._modules = ModuleList(modules)

    def forward(self, x: Any) -> Any:
        for module in self._modules:
            x = module(x)
        return x

    def __getitem__(self, index: int) -> _FrontendModule:
        return self._modules[index]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[_FrontendModule]:
        return iter(self._modules)


def _iter_named_parameters(
    module: _FrontendModule,
    prefix: str,
    recurse: bool,
    seen: set[int],
) -> Iterator[tuple[str, Parameter]]:
    if isinstance(module, (ModuleList, Sequential)):
        for index, child in enumerate(module):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            yield from _iter_named_parameters(child, child_prefix, recurse, seen)
        return
    for name, value in vars(module).items():
        if name.startswith("_"):
            continue
        child_prefix = f"{prefix}.{name}" if prefix else name
        if isinstance(value, Parameter):
            if id(value) not in seen:
                seen.add(id(value))
                yield child_prefix, value
        elif recurse and isinstance(value, _FrontendModule):
            yield from _iter_named_parameters(value, child_prefix, recurse, seen)
        elif recurse and isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if isinstance(item, _FrontendModule):
                    yield from _iter_named_parameters(item, f"{child_prefix}.{index}", recurse, seen)


def _set_auto_parameter_name(parameter: Parameter, name: str) -> None:
    parameter.name = name
    setattr(parameter, "_dinoml_auto_name", name)


def _prefix_auto_parameter_names(module: _FrontendModule, prefix: str) -> None:
    object.__setattr__(module, "_dinoml_auto_prefix", prefix)
    for value in vars(module).values():
        if isinstance(value, Parameter):
            auto_name = getattr(value, "_dinoml_auto_name", None)
            if auto_name:
                _set_auto_parameter_name(value, f"{prefix}_{auto_name}")
        elif isinstance(value, _FrontendModule):
            _prefix_auto_parameter_names(value, prefix)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, _FrontendModule):
                    _prefix_auto_parameter_names(item, prefix)


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _single_normalized_shape(value: int | Sequence[int], name: str) -> int:
    normalized = _single(value, name, positive=True)
    return normalized[0]


def _single(
    value: int | Sequence[int],
    name: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> tuple[int]:
    values = _sequence_of_ints(value, 1, name)
    _validate_numeric_bounds(values, name, positive=positive, non_negative=non_negative)
    return (values[0],)


def _pair(
    value: int | Sequence[int],
    name: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> tuple[int, int]:
    values = _sequence_of_ints(value, 2, name)
    _validate_numeric_bounds(values, name, positive=positive, non_negative=non_negative)
    return values[0], values[1]


def _triple(
    value: int | Sequence[int],
    name: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> tuple[int, int, int]:
    values = _sequence_of_ints(value, 3, name)
    _validate_numeric_bounds(values, name, positive=positive, non_negative=non_negative)
    return values[0], values[1], values[2]


def _sequence_of_ints(value: int | Sequence[int], length: int, name: str) -> tuple[int, ...]:
    if isinstance(value, int) and not isinstance(value, bool):
        return tuple(int(value) for _ in range(length))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = tuple(value)
        if len(values) != length:
            raise ValueError(f"{name} must be an integer or a length-{length} sequence")
        if any(not isinstance(item, int) or isinstance(item, bool) for item in values):
            raise TypeError(f"{name} must contain integers")
        return tuple(int(item) for item in values)
    raise TypeError(f"{name} must be an integer or a length-{length} sequence")


def _validate_numeric_bounds(
    values: tuple[int, ...],
    name: str,
    *,
    positive: bool,
    non_negative: bool,
) -> None:
    if positive and any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integers")
    if non_negative and any(value < 0 for value in values):
        raise ValueError(f"{name} must contain non-negative integers")


def _upsample_scale_factor_from_size(x: Any, size: int | Sequence[int]) -> float:
    tensor = as_tensor(x)
    if tensor.rank not in (3, 4, 5):
        raise ValueError(f"Upsample expects rank-3, rank-4, or rank-5 dense tensors, got rank {tensor.rank}")
    spatial_rank = tensor.rank - 2
    target_sizes = _upsample_output_size(size, spatial_rank)
    input_sizes = tensor.shape_spec[2:]
    if any(not isinstance(dim, int) for dim in input_sizes):
        raise NotImplementedError("Upsample size= currently requires static spatial input dimensions")
    normalized_input_sizes = tuple(int(dim) for dim in input_sizes)
    scale_factors = tuple(float(target) / float(source) for source, target in zip(normalized_input_sizes, target_sizes))
    first = scale_factors[0]
    if any(value <= 0.0 for value in scale_factors):
        raise ValueError("Upsample size= must produce positive output extents")
    if any(value != first for value in scale_factors[1:]):
        raise NotImplementedError("Upsample size= currently requires a uniform scale_factor across spatial dims")
    return first


def _upsample_output_size(size: int | Sequence[int], spatial_rank: int) -> tuple[int, ...]:
    name = "Upsample size"
    if isinstance(size, int) and not isinstance(size, bool):
        if size <= 0:
            raise ValueError(f"{name} must contain positive integers")
        return tuple(int(size) for _ in range(spatial_rank))
    if isinstance(size, Sequence) and not isinstance(size, (str, bytes, bytearray)):
        values = tuple(size)
        if len(values) != spatial_rank:
            raise ValueError(f"{name} must be an integer or a length-{spatial_rank} sequence")
        if any(not isinstance(item, int) or isinstance(item, bool) for item in values):
            raise TypeError(f"{name} must contain integers")
        normalized = tuple(int(item) for item in values)
        if any(item <= 0 for item in normalized):
            raise ValueError(f"{name} must contain positive integers")
        return normalized
    raise TypeError(f"{name} must be an integer or a length-{spatial_rank} sequence")


__all__ = [
    "AvgPool1d",
    "AvgPool2d",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "Dropout",
    "Embedding",
    "Flatten",
    "GELU",
    "GroupNorm",
    "Identity",
    "LayerNorm",
    "Linear",
    "MaxPool2d",
    "Module",
    "ModuleList",
    "Parameter",
    "RMSNorm",
    "ReLU",
    "Sequential",
    "SiLU",
    "Sigmoid",
    "Softmax",
    "T5LayerNorm",
    "Tanh",
    "Unflatten",
    "Upsampling1d",
    "Upsampling1dAdd",
    "Upsampling2d",
    "Upsampling2dAdd",
    "Upsampling3d",
    "Upsampling3dAdd",
    "Upsampling3dCompressTime",
    "Upsample",
    "functional",
]
