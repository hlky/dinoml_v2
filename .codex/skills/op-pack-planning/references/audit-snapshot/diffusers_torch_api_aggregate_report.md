# Torch API Usage by Diffusers Model Components

## torch and torch.Tensor

| function | count |
| --- | ---: |
| `to` | 91 |
| `cat` | 90 |
| `reshape` | 79 |
| `is_grad_enabled` | 78 |
| `unsqueeze` | 74 |
| `permute` | 69 |
| `chunk` | 61 |
| `flatten` | 59 |
| `view` | 58 |
| `float` | 48 |
| `arange` | 44 |
| `unflatten` | 43 |
| `zeros` | 41 |
| `expand` | 39 |
| `tensor` | 37 |
| `transpose` | 37 |
| `split` | 36 |
| `size` | 33 |
| `contiguous` | 32 |
| `ones` | 30 |
| `repeat_interleave` | 30 |
| `mean` | 27 |
| `randn` | 27 |
| `type_as` | 26 |
| `repeat` | 24 |
| `sin` | 24 |
| `mode` | 23 |
| `squeeze` | 23 |
| `cos` | 22 |
| `stack` | 22 |
| `zeros_like` | 18 |
| `meshgrid` | 17 |
| `sum` | 17 |
| `unbind` | 17 |
| `concat` | 15 |
| `einsum` | 15 |
| `outer` | 14 |
| `clone` | 13 |
| `is_tensor` | 13 |
| `bool` | 12 |
| `where` | 12 |
| `clip` | 11 |
| `new_zeros` | 11 |
| `ones_like` | 11 |
| `exp` | 10 |
| `tanh` | 10 |
| `pow` | 9 |
| `clamp` | 8 |
| `empty_like` | 8 |
| `device` | 7 |
| `dim` | 7 |
| `flip` | 7 |
| `no_grad` | 7 |
| `sigmoid` | 7 |
| `polar` | 6 |
| `softmax` | 6 |
| `split_with_sizes` | 6 |
| `div` | 5 |
| `empty` | 5 |
| `full` | 5 |
| `item` | 5 |
| `linspace` | 5 |
| `max` | 5 |
| `movedim` | 5 |
| `mul` | 5 |
| `numel` | 5 |
| `view_as_complex` | 5 |
| `view_as_real` | 5 |
| `Tensor` | 4 |
| `concatenate` | 4 |
| `gather` | 4 |
| `matmul` | 4 |
| `norm` | 4 |
| `tolist` | 4 |
| `add` | 3 |
| `all` | 3 |
| `any` | 3 |
| `broadcast_to` | 3 |
| `cumsum` | 3 |
| `detach` | 3 |
| `finfo` | 3 |
| `log` | 3 |
| `logspace` | 3 |
| `masked_fill` | 3 |
| `new_ones` | 3 |
| `type` | 3 |
| `argsort` | 2 |
| `as_tensor` | 2 |
| `autocast` | 2 |
| `copy` | 2 |
| `cpu` | 2 |
| `from_numpy` | 2 |
| `long` | 2 |
| `rand` | 2 |
| `resize` | 2 |
| `round` | 2 |
| `rsqrt` | 2 |
| `scatter_add` | 2 |
| `sqrt` | 2 |
| `sub` | 2 |
| `topk` | 2 |
| `values` | 2 |
| `Size` | 1 |
| `abs` | 1 |
| `addcmul` | 1 |
| `argmax` | 1 |
| `argmin` | 1 |
| `atanh` | 1 |
| `baddbmm` | 1 |
| `bincount` | 1 |
| `bmm` | 1 |
| `byte` | 1 |
| `cdist` | 1 |
| `cumprod` | 1 |
| `diag_embed` | 1 |
| `double` | 1 |
| `element_size` | 1 |
| `full_like` | 1 |
| `int` | 1 |
| `lerp` | 1 |
| `logical_not` | 1 |
| `logical_or` | 1 |
| `maximum` | 1 |
| `numpy` | 1 |
| `randint` | 1 |
| `reciprocal` | 1 |
| `relu` | 1 |
| `requires_grad` | 1 |
| `scatter_reduce` | 1 |
| `sort` | 1 |
| `swapaxes` | 1 |
| `tril` | 1 |
| `triu` | 1 |
| `vstack` | 1 |
| `zero` | 1 |

## torch.nn

| module | count |
| --- | ---: |
| `ModuleList` | 107 |
| `Linear` | 82 |
| `SiLU` | 56 |
| `LayerNorm` | 52 |
| `Parameter` | 46 |
| `Conv2d` | 41 |
| `Dropout` | 37 |
| `Sequential` | 28 |
| `Conv3d` | 25 |
| `GroupNorm` | 25 |
| `Identity` | 25 |
| `GELU` | 19 |
| `RMSNorm` | 19 |
| `Embedding` | 16 |
| `Conv1d` | 11 |
| `ConvTranspose1d` | 4 |
| `ConvTranspose2d` | 4 |
| `Module` | 4 |
| `PixelUnshuffle` | 4 |
| `PixelShuffle` | 3 |
| `AvgPool2d` | 2 |
| `AvgPool3d` | 2 |
| `BatchNorm2d` | 2 |
| `ModuleDict` | 2 |
| `ReLU` | 2 |
| `Upsample` | 2 |
| `ZeroPad2d` | 2 |
| `AdaptiveAvgPool2d` | 1 |
| `AvgPool1d` | 1 |
| `ELU` | 1 |
| `LeakyReLU` | 1 |
| `MultiheadAttention` | 1 |
| `Tanh` | 1 |

## torch.nn.functional

| function | count |
| --- | ---: |
| `pad` | 26 |
| `silu` | 18 |
| `interpolate` | 12 |
| `scaled_dot_product_attention` | 12 |
| `normalize` | 5 |
| `avg_pool3d` | 3 |
| `conv2d` | 3 |
| `avg_pool1d` | 2 |
| `gelu` | 2 |
| `layer_norm` | 2 |
| `linear` | 2 |
| `softmax` | 2 |
| `avg_pool2d` | 1 |
| `conv1d` | 1 |
| `conv3d` | 1 |
| `conv_transpose1d` | 1 |
| `conv_transpose2d` | 1 |
| `conv_transpose3d` | 1 |
| `embedding` | 1 |
| `group_norm` | 1 |
| `grouped_mm` | 1 |
| `leaky_relu` | 1 |
| `log_softmax` | 1 |
| `max_pool3d` | 1 |
| `multi_head_attention_forward` | 1 |
| `one_hot` | 1 |
| `pixel_shuffle` | 1 |
| `pixel_unshuffle` | 1 |
| `relu` | 1 |
| `softplus` | 1 |

## Alphabetical torch and torch.Tensor Functions

- `Size`
- `Tensor`
- `abs`
- `add`
- `addcmul`
- `all`
- `any`
- `arange`
- `argmax`
- `argmin`
- `argsort`
- `as_tensor`
- `atanh`
- `autocast`
- `baddbmm`
- `bincount`
- `bmm`
- `bool`
- `broadcast_to`
- `byte`
- `cat`
- `cdist`
- `chunk`
- `clamp`
- `clip`
- `clone`
- `concat`
- `concatenate`
- `contiguous`
- `copy`
- `cos`
- `cpu`
- `cumprod`
- `cumsum`
- `detach`
- `device`
- `diag_embed`
- `dim`
- `div`
- `double`
- `einsum`
- `element_size`
- `empty`
- `empty_like`
- `exp`
- `expand`
- `finfo`
- `flatten`
- `flip`
- `float`
- `from_numpy`
- `full`
- `full_like`
- `gather`
- `int`
- `is_grad_enabled`
- `is_tensor`
- `item`
- `lerp`
- `linspace`
- `log`
- `logical_not`
- `logical_or`
- `logspace`
- `long`
- `masked_fill`
- `matmul`
- `max`
- `maximum`
- `mean`
- `meshgrid`
- `mode`
- `movedim`
- `mul`
- `new_ones`
- `new_zeros`
- `no_grad`
- `norm`
- `numel`
- `numpy`
- `ones`
- `ones_like`
- `outer`
- `permute`
- `polar`
- `pow`
- `rand`
- `randint`
- `randn`
- `reciprocal`
- `relu`
- `repeat`
- `repeat_interleave`
- `requires_grad`
- `reshape`
- `resize`
- `round`
- `rsqrt`
- `scatter_add`
- `scatter_reduce`
- `sigmoid`
- `sin`
- `size`
- `softmax`
- `sort`
- `split`
- `split_with_sizes`
- `sqrt`
- `squeeze`
- `stack`
- `sub`
- `sum`
- `swapaxes`
- `tanh`
- `tensor`
- `to`
- `tolist`
- `topk`
- `transpose`
- `tril`
- `triu`
- `type`
- `type_as`
- `unbind`
- `unflatten`
- `unsqueeze`
- `values`
- `view`
- `view_as_complex`
- `view_as_real`
- `vstack`
- `where`
- `zero`
- `zeros`
- `zeros_like`

## Alphabetical torch.nn Modules

- `AdaptiveAvgPool2d`
- `AvgPool1d`
- `AvgPool2d`
- `AvgPool3d`
- `BatchNorm2d`
- `Conv1d`
- `Conv2d`
- `Conv3d`
- `ConvTranspose1d`
- `ConvTranspose2d`
- `Dropout`
- `ELU`
- `Embedding`
- `GELU`
- `GroupNorm`
- `Identity`
- `LayerNorm`
- `LeakyReLU`
- `Linear`
- `Module`
- `ModuleDict`
- `ModuleList`
- `MultiheadAttention`
- `Parameter`
- `PixelShuffle`
- `PixelUnshuffle`
- `RMSNorm`
- `ReLU`
- `Sequential`
- `SiLU`
- `Tanh`
- `Upsample`
- `ZeroPad2d`

## Alphabetical torch.nn.functional Functions

- `avg_pool1d`
- `avg_pool2d`
- `avg_pool3d`
- `conv1d`
- `conv2d`
- `conv3d`
- `conv_transpose1d`
- `conv_transpose2d`
- `conv_transpose3d`
- `embedding`
- `gelu`
- `group_norm`
- `grouped_mm`
- `interpolate`
- `layer_norm`
- `leaky_relu`
- `linear`
- `log_softmax`
- `max_pool3d`
- `multi_head_attention_forward`
- `normalize`
- `one_hot`
- `pad`
- `pixel_shuffle`
- `pixel_unshuffle`
- `relu`
- `scaled_dot_product_attention`
- `silu`
- `softmax`
- `softplus`
