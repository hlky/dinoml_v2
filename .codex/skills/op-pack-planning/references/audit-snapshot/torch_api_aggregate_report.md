# Torch API Usage by Transformers Model Family

## torch and torch.Tensor

| function | count |
| --- | ---: |
| `view` | 388 |
| `transpose` | 383 |
| `contiguous` | 374 |
| `reshape` | 373 |
| `to` | 371 |
| `no_grad` | 361 |
| `unsqueeze` | 360 |
| `arange` | 347 |
| `matmul` | 345 |
| `cat` | 344 |
| `expand` | 290 |
| `zeros` | 285 |
| `size` | 284 |
| `ones` | 279 |
| `permute` | 270 |
| `sum` | 247 |
| `float` | 235 |
| `mean` | 231 |
| `clone` | 211 |
| `sin` | 205 |
| `cos` | 204 |
| `copy` | 196 |
| `normal` | 196 |
| `tensor` | 194 |
| `flatten` | 181 |
| `squeeze` | 176 |
| `pow` | 163 |
| `clamp` | 160 |
| `rsqrt` | 153 |
| `split` | 147 |
| `where` | 142 |
| `masked_fill` | 139 |
| `stack` | 132 |
| `zeros_like` | 125 |
| `cumsum` | 123 |
| `all` | 120 |
| `repeat` | 119 |
| `max` | 103 |
| `type_as` | 101 |
| `finfo` | 97 |
| `gather` | 97 |
| `empty` | 96 |
| `rand` | 96 |
| `expand_as` | 91 |
| `chunk` | 89 |
| `div` | 88 |
| `long` | 88 |
| `numel` | 80 |
| `bool` | 75 |
| `randn` | 74 |
| `topk` | 72 |
| `exp` | 71 |
| `new_zeros` | 71 |
| `ones_like` | 69 |
| `detach` | 68 |
| `repeat_interleave` | 68 |
| `item` | 67 |
| `masked_scatter` | 66 |
| `dim` | 64 |
| `log` | 63 |
| `sigmoid` | 63 |
| `int` | 62 |
| `nonzero` | 61 |
| `argmax` | 60 |
| `tolist` | 55 |
| `uniform` | 54 |
| `einsum` | 53 |
| `floor` | 51 |
| `full` | 50 |
| `abs` | 48 |
| `ne` | 47 |
| `tanh` | 47 |
| `index_add` | 44 |
| `softmax` | 44 |
| `linspace` | 42 |
| `full_like` | 40 |
| `unbind` | 40 |
| `any` | 39 |
| `greater` | 39 |
| `type` | 38 |
| `bmm` | 35 |
| `min` | 35 |
| `prod` | 35 |
| `meshgrid` | 33 |
| `outer` | 29 |
| `scatter` | 29 |
| `Tensor` | 28 |
| `flip` | 28 |
| `get_default_dtype` | 26 |
| `isfinite` | 26 |
| `t` | 26 |
| `tril` | 26 |
| `sqrt` | 24 |
| `new_ones` | 23 |
| `device` | 21 |
| `requires_grad` | 20 |
| `concat` | 18 |
| `clip` | 17 |
| `empty_like` | 17 |
| `eq` | 17 |
| `FloatTensor` | 16 |
| `as_tensor` | 16 |
| `values` | 16 |
| `roll` | 15 |
| `eye` | 14 |
| `masked_select` | 14 |
| `norm` | 14 |
| `argsort` | 13 |
| `cpu` | 13 |
| `index_select` | 13 |
| `triu` | 13 |
| `unfold` | 12 |
| `mul` | 11 |
| `tile` | 11 |
| `unique_consecutive` | 11 |
| `ceil` | 10 |
| `index` | 10 |
| `std` | 10 |
| `expm1` | 9 |
| `get_autocast_dtype` | 9 |
| `is_autocast_enabled` | 9 |
| `isinf` | 9 |
| `maximum` | 9 |
| `rand_like` | 9 |
| `argmin` | 8 |
| `clamp_min` | 8 |
| `concatenate` | 8 |
| `reshape_as` | 8 |
| `fill` | 7 |
| `minimum` | 7 |
| `new_full` | 7 |
| `add` | 6 |
| `baddbmm` | 6 |
| `diff` | 6 |
| `log1p` | 6 |
| `mm` | 6 |
| `randint_like` | 6 |
| `sign` | 6 |
| `sort` | 6 |
| `broadcast_to` | 5 |
| `bucketize` | 5 |
| `isin` | 5 |
| `logical_not` | 5 |
| `logsumexp` | 5 |
| `nan_to_num` | 5 |
| `narrow` | 5 |
| `round` | 5 |
| `searchsorted` | 5 |
| `square` | 5 |
| `xlogy` | 5 |
| `bernoulli` | 4 |
| `cosine_similarity` | 4 |
| `inference_mode` | 4 |
| `log2` | 4 |
| `logical_and` | 4 |
| `multinomial` | 4 |
| `relu` | 4 |
| `unique` | 4 |
| `var` | 4 |
| `view_as` | 4 |
| `amax` | 3 |
| `is_tensor` | 3 |
| `new` | 3 |
| `new_tensor` | 3 |
| `numpy` | 3 |
| `randn_like` | 3 |
| `register_hook` | 3 |
| `LongTensor` | 2 |
| `Size` | 2 |
| `argwhere` | 2 |
| `as_strided` | 2 |
| `cummax` | 2 |
| `cummin` | 2 |
| `erf` | 2 |
| `ge` | 2 |
| `is_contiguous` | 2 |
| `is_floating_point` | 2 |
| `load` | 2 |
| `lt` | 2 |
| `movedim` | 2 |
| `multiply` | 2 |
| `polar` | 2 |
| `put` | 2 |
| `randint` | 2 |
| `randperm` | 2 |
| `reciprocal` | 2 |
| `scatter_add` | 2 |
| `scatter_reduce` | 2 |
| `stride` | 2 |
| `sub` | 2 |
| `take_along_dim` | 2 |
| `view_as_complex` | 2 |
| `view_as_real` | 2 |
| `IntTensor` | 1 |
| `acos` | 1 |
| `amin` | 1 |
| `are_deterministic_algorithms_enabled` | 1 |
| `asinh` | 1 |
| `backward` | 1 |
| `bfloat16` | 1 |
| `bincount` | 1 |
| `bitwise_xor` | 1 |
| `broadcast_tensors` | 1 |
| `cdist` | 1 |
| `clamp_max` | 1 |
| `conj` | 1 |
| `cross` | 1 |
| `diagonal` | 1 |
| `enable_grad` | 1 |
| `exponential` | 1 |
| `fill_diagonal` | 1 |
| `floor_divide` | 1 |
| `fmod` | 1 |
| `from_numpy` | 1 |
| `histc` | 1 |
| `hstack` | 1 |
| `iinfo` | 1 |
| `index_copy` | 1 |
| `index_put` | 1 |
| `isnan` | 1 |
| `kaiser_window` | 1 |
| `log10` | 1 |
| `log_softmax` | 1 |
| `logical_or` | 1 |
| `manual_seed` | 1 |
| `ndimension` | 1 |
| `neg` | 1 |
| `remainder` | 1 |
| `scalar_tensor` | 1 |
| `seed` | 1 |
| `sinc` | 1 |
| `split_with_sizes` | 1 |
| `tensor_split` | 1 |
| `unflatten` | 1 |
| `unravel_index` | 1 |
| `zero` | 1 |

## torch.nn

| module | count |
| --- | ---: |
| `Linear` | 425 |
| `ModuleList` | 405 |
| `Parameter` | 310 |
| `Embedding` | 291 |
| `LayerNorm` | 271 |
| `Dropout` | 198 |
| `Conv2d` | 154 |
| `CrossEntropyLoss` | 141 |
| `Identity` | 121 |
| `BCEWithLogitsLoss` | 81 |
| `MSELoss` | 76 |
| `Conv1d` | 64 |
| `ReLU` | 58 |
| `Sequential` | 55 |
| `Tanh` | 52 |
| `GELU` | 49 |
| `BatchNorm2d` | 43 |
| `GroupNorm` | 26 |
| `ConvTranspose2d` | 25 |
| `AdaptiveAvgPool2d` | 21 |
| `Softmax` | 21 |
| `Conv3d` | 16 |
| `BatchNorm1d` | 15 |
| `ConvTranspose1d` | 14 |
| `MaxPool2d` | 14 |
| `Sigmoid` | 13 |
| `SiLU` | 11 |
| `AvgPool2d` | 10 |
| `ModuleDict` | 10 |
| `AdaptiveAvgPool1d` | 8 |
| `Flatten` | 8 |
| `MultiheadAttention` | 8 |
| `Upsample` | 8 |
| `AvgPool1d` | 7 |
| `ELU` | 7 |
| `GLU` | 6 |
| `PixelShuffle` | 6 |
| `Dropout2d` | 4 |
| `ZeroPad2d` | 4 |
| `LogSoftmax` | 3 |
| `Module` | 3 |
| `AdaptiveLogSoftmaxWithLoss` | 2 |
| `GRUCell` | 2 |
| `Hardsigmoid` | 2 |
| `L1Loss` | 2 |
| `MaxPool1d` | 2 |
| `ParameterList` | 2 |
| `Softplus` | 2 |
| `Unfold` | 2 |
| `BatchNorm3d` | 1 |
| `ConstantPad1d` | 1 |
| `ConstantPad2d` | 1 |
| `Hardswish` | 1 |
| `KLDivLoss` | 1 |
| `LSTM` | 1 |
| `LeakyReLU` | 1 |
| `ParameterDict` | 1 |
| `RMSNorm` | 1 |
| `ReflectionPad2d` | 1 |
| `SmoothL1Loss` | 1 |
| `SyncBatchNorm` | 1 |

## torch.nn.functional

| function | count |
| --- | ---: |
| `pad` | 76 |
| `softmax` | 64 |
| `one_hot` | 45 |
| `interpolate` | 38 |
| `linear` | 38 |
| `sigmoid` | 20 |
| `softplus` | 16 |
| `scaled_dot_product_attention` | 13 |
| `dropout` | 9 |
| `grid_sample` | 8 |
| `normalize` | 8 |
| `silu` | 8 |
| `conv1d` | 7 |
| `relu` | 7 |
| `layer_norm` | 6 |
| `logsigmoid` | 5 |
| `embedding` | 4 |
| `mse_loss` | 4 |
| `unfold` | 4 |
| `adaptive_avg_pool2d` | 2 |
| `avg_pool2d` | 2 |
| `conv_transpose1d` | 1 |
| `gelu` | 1 |
| `glu` | 1 |
| `group_norm` | 1 |
| `multi_head_attention_forward` | 1 |
| `smooth_l1_loss` | 1 |

## Alphabetical torch and torch.Tensor Functions

- `FloatTensor`
- `IntTensor`
- `LongTensor`
- `Size`
- `Tensor`
- `abs`
- `acos`
- `add`
- `all`
- `amax`
- `amin`
- `any`
- `arange`
- `are_deterministic_algorithms_enabled`
- `argmax`
- `argmin`
- `argsort`
- `argwhere`
- `as_strided`
- `as_tensor`
- `asinh`
- `backward`
- `baddbmm`
- `bernoulli`
- `bfloat16`
- `bincount`
- `bitwise_xor`
- `bmm`
- `bool`
- `broadcast_tensors`
- `broadcast_to`
- `bucketize`
- `cat`
- `cdist`
- `ceil`
- `chunk`
- `clamp`
- `clamp_max`
- `clamp_min`
- `clip`
- `clone`
- `concat`
- `concatenate`
- `conj`
- `contiguous`
- `copy`
- `cos`
- `cosine_similarity`
- `cpu`
- `cross`
- `cummax`
- `cummin`
- `cumsum`
- `detach`
- `device`
- `diagonal`
- `diff`
- `dim`
- `div`
- `einsum`
- `empty`
- `empty_like`
- `enable_grad`
- `eq`
- `erf`
- `exp`
- `expand`
- `expand_as`
- `expm1`
- `exponential`
- `eye`
- `fill`
- `fill_diagonal`
- `finfo`
- `flatten`
- `flip`
- `float`
- `floor`
- `floor_divide`
- `fmod`
- `from_numpy`
- `full`
- `full_like`
- `gather`
- `ge`
- `get_autocast_dtype`
- `get_default_dtype`
- `greater`
- `histc`
- `hstack`
- `iinfo`
- `index`
- `index_add`
- `index_copy`
- `index_put`
- `index_select`
- `inference_mode`
- `int`
- `is_autocast_enabled`
- `is_contiguous`
- `is_floating_point`
- `is_tensor`
- `isfinite`
- `isin`
- `isinf`
- `isnan`
- `item`
- `kaiser_window`
- `linspace`
- `load`
- `log`
- `log10`
- `log1p`
- `log2`
- `log_softmax`
- `logical_and`
- `logical_not`
- `logical_or`
- `logsumexp`
- `long`
- `lt`
- `manual_seed`
- `masked_fill`
- `masked_scatter`
- `masked_select`
- `matmul`
- `max`
- `maximum`
- `mean`
- `meshgrid`
- `min`
- `minimum`
- `mm`
- `movedim`
- `mul`
- `multinomial`
- `multiply`
- `nan_to_num`
- `narrow`
- `ndimension`
- `ne`
- `neg`
- `new`
- `new_full`
- `new_ones`
- `new_tensor`
- `new_zeros`
- `no_grad`
- `nonzero`
- `norm`
- `normal`
- `numel`
- `numpy`
- `ones`
- `ones_like`
- `outer`
- `permute`
- `polar`
- `pow`
- `prod`
- `put`
- `rand`
- `rand_like`
- `randint`
- `randint_like`
- `randn`
- `randn_like`
- `randperm`
- `reciprocal`
- `register_hook`
- `relu`
- `remainder`
- `repeat`
- `repeat_interleave`
- `requires_grad`
- `reshape`
- `reshape_as`
- `roll`
- `round`
- `rsqrt`
- `scalar_tensor`
- `scatter`
- `scatter_add`
- `scatter_reduce`
- `searchsorted`
- `seed`
- `sigmoid`
- `sign`
- `sin`
- `sinc`
- `size`
- `softmax`
- `sort`
- `split`
- `split_with_sizes`
- `sqrt`
- `square`
- `squeeze`
- `stack`
- `std`
- `stride`
- `sub`
- `sum`
- `t`
- `take_along_dim`
- `tanh`
- `tensor`
- `tensor_split`
- `tile`
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
- `unfold`
- `uniform`
- `unique`
- `unique_consecutive`
- `unravel_index`
- `unsqueeze`
- `values`
- `var`
- `view`
- `view_as`
- `view_as_complex`
- `view_as_real`
- `where`
- `xlogy`
- `zero`
- `zeros`
- `zeros_like`

## Alphabetical torch.nn Modules

- `AdaptiveAvgPool1d`
- `AdaptiveAvgPool2d`
- `AdaptiveLogSoftmaxWithLoss`
- `AvgPool1d`
- `AvgPool2d`
- `BCEWithLogitsLoss`
- `BatchNorm1d`
- `BatchNorm2d`
- `BatchNorm3d`
- `ConstantPad1d`
- `ConstantPad2d`
- `Conv1d`
- `Conv2d`
- `Conv3d`
- `ConvTranspose1d`
- `ConvTranspose2d`
- `CrossEntropyLoss`
- `Dropout`
- `Dropout2d`
- `ELU`
- `Embedding`
- `Flatten`
- `GELU`
- `GLU`
- `GRUCell`
- `GroupNorm`
- `Hardsigmoid`
- `Hardswish`
- `Identity`
- `KLDivLoss`
- `L1Loss`
- `LSTM`
- `LayerNorm`
- `LeakyReLU`
- `Linear`
- `LogSoftmax`
- `MSELoss`
- `MaxPool1d`
- `MaxPool2d`
- `Module`
- `ModuleDict`
- `ModuleList`
- `MultiheadAttention`
- `Parameter`
- `ParameterDict`
- `ParameterList`
- `PixelShuffle`
- `RMSNorm`
- `ReLU`
- `ReflectionPad2d`
- `Sequential`
- `SiLU`
- `Sigmoid`
- `SmoothL1Loss`
- `Softmax`
- `Softplus`
- `SyncBatchNorm`
- `Tanh`
- `Unfold`
- `Upsample`
- `ZeroPad2d`

## Alphabetical torch.nn.functional Functions

- `adaptive_avg_pool2d`
- `avg_pool2d`
- `conv1d`
- `conv_transpose1d`
- `dropout`
- `embedding`
- `gelu`
- `glu`
- `grid_sample`
- `group_norm`
- `interpolate`
- `layer_norm`
- `linear`
- `logsigmoid`
- `mse_loss`
- `multi_head_attention_forward`
- `normalize`
- `one_hot`
- `pad`
- `relu`
- `scaled_dot_product_attention`
- `sigmoid`
- `silu`
- `smooth_l1_loss`
- `softmax`
- `softplus`
- `unfold`
