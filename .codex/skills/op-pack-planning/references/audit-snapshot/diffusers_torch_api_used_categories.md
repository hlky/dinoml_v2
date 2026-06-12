# Categorized Public Torch APIs Used by Diffusers Model Components

Source: `X:\H\diffusers\src\diffusers`

Used means present in the selected Diffusers model component files. Function names are normalized so in-place suffixes such as `_` are folded into the base name.

## torch and torch.Tensor functions (135)

### Devices, dtypes, autocast, RNG, and runtime configuration (1)

`is_grad_enabled`

### Dispatch, overrides, predicates, and introspection (2)

`is_tensor`, `requires_grad`

### Dtype conversion and tensor property checks (3)

`byte`, `double`, `element_size`

### Elementwise math, comparisons, and special functions (2)

`atanh`, `lerp`

### Miscellaneous low-level helpers and aliases (120)

`Size`, `Tensor`, `abs`, `add`, `addcmul`, `all`, `any`, `arange`, `argmax`, `argmin`, `argsort`, `as_tensor`, `autocast`, `baddbmm`, `bincount`, `bmm`, `bool`, `broadcast_to`, `cat`, `cdist`, `chunk`, `clamp`, `clip`, `clone`, `concat`, `concatenate`, `contiguous`, `copy`, `cos`, `cpu`, `cumsum`, `detach`, `device`, `dim`, `div`, `einsum`, `empty`, `empty_like`, `exp`, `expand`, `finfo`, `flatten`, `flip`, `float`, `from_numpy`, `full`, `full_like`, `gather`, `int`, `item`, `linspace`, `log`, `logical_not`, `logical_or`, `logspace`, `long`, `masked_fill`, `matmul`, `max`, `maximum`, `mean`, `meshgrid`, `movedim`, `mul`, `new_ones`, `new_zeros`, `no_grad`, `norm`, `numel`, `numpy`, `ones`, `ones_like`, `outer`, `permute`, `polar`, `pow`, `rand`, `randint`, `randn`, `reciprocal`, `repeat`, `repeat_interleave`, `reshape`, `round`, `rsqrt`, `scatter_add`, `scatter_reduce`, `sigmoid`, `sin`, `size`, `softmax`, `sort`, `split`, `split_with_sizes`, `sqrt`, `squeeze`, `stack`, `sub`, `sum`, `tanh`, `tensor`, `to`, `tolist`, `topk`, `transpose`, `tril`, `triu`, `type`, `type_as`, `unbind`, `unflatten`, `unsqueeze`, `values`, `view`, `view_as_complex`, `view_as_real`, `where`, `zero`, `zeros`, `zeros_like`

### Neural network ops, activations, losses, and fused kernels (1)

`relu`

### Reductions, statistics, and numerical analysis (2)

`cumprod`, `mode`

### Tensor construction, shape, views, indexing, and copies (4)

`diag_embed`, `resize`, `swapaxes`, `vstack`

## torch.nn modules (33)

### Convolution, pooling, padding, and spatial reshaping modules (12)

`AdaptiveAvgPool2d`, `AvgPool1d`, `AvgPool2d`, `AvgPool3d`, `Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose1d`, `ConvTranspose2d`, `PixelShuffle`, `PixelUnshuffle`, `ZeroPad2d`

### Dropout modules (1)

`Dropout`

### Normalization modules (4)

`BatchNorm2d`, `GroupNorm`, `LayerNorm`, `RMSNorm`

### Other nn modules (16)

`ELU`, `Embedding`, `GELU`, `Identity`, `LeakyReLU`, `Linear`, `Module`, `ModuleDict`, `ModuleList`, `MultiheadAttention`, `Parameter`, `ReLU`, `Sequential`, `SiLU`, `Tanh`, `Upsample`

## torch.nn.functional functions (30)

### Activation and probability transform functions (2)

`leaky_relu`, `log_softmax`

### Convolution, bilinear, and matrix kernels (7)

`conv1d`, `conv2d`, `conv3d`, `conv_transpose1d`, `conv_transpose2d`, `conv_transpose3d`, `grouped_mm`

### Normalization functions (3)

`group_norm`, `layer_norm`, `normalize`

### Other functional APIs (12)

`embedding`, `gelu`, `interpolate`, `linear`, `multi_head_attention_forward`, `one_hot`, `pad`, `relu`, `scaled_dot_product_attention`, `silu`, `softmax`, `softplus`

### Pooling and unpooling functions (4)

`avg_pool1d`, `avg_pool2d`, `avg_pool3d`, `max_pool3d`

### Spatial transform, channel, and pixel layout functions (2)

`pixel_shuffle`, `pixel_unshuffle`
