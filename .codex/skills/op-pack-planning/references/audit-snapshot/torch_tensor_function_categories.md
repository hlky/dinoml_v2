# torch and torch.Tensor Function Categories

These categories cover the normalized aggregate list from `X:\H\transformers`.
Each function appears exactly once.

## Tensor constructors and factories (28)

`FloatTensor`, `IntTensor`, `LongTensor`, `Size`, `Tensor`, `arange`, `as_tensor`, `empty`, `empty_like`, `eye`, `from_numpy`, `full`, `full_like`, `kaiser_window`, `linspace`, `meshgrid`, `new`, `new_full`, `new_ones`, `new_tensor`, `new_zeros`, `ones`, `ones_like`, `polar`, `scalar_tensor`, `tensor`, `zeros`, `zeros_like`

## Dtype, device, conversion, and scalar extraction (18)

`bfloat16`, `bool`, `copy`, `cpu`, `device`, `finfo`, `float`, `get_autocast_dtype`, `get_default_dtype`, `iinfo`, `int`, `item`, `long`, `numpy`, `to`, `tolist`, `type`, `type_as`

## Randomness and initialization (17)

`bernoulli`, `exponential`, `fill`, `fill_diagonal`, `manual_seed`, `multinomial`, `normal`, `rand`, `rand_like`, `randint`, `randint_like`, `randn`, `randn_like`, `randperm`, `seed`, `uniform`, `zero`

## Shape, view, layout, and concatenation (44)

`as_strided`, `broadcast_tensors`, `broadcast_to`, `cat`, `chunk`, `clone`, `concat`, `concatenate`, `contiguous`, `dim`, `expand`, `expand_as`, `flatten`, `flip`, `hstack`, `movedim`, `narrow`, `ndimension`, `numel`, `permute`, `repeat`, `repeat_interleave`, `reshape`, `reshape_as`, `roll`, `size`, `split`, `split_with_sizes`, `squeeze`, `stack`, `stride`, `t`, `tensor_split`, `tile`, `transpose`, `unbind`, `unflatten`, `unfold`, `unravel_index`, `unsqueeze`, `view`, `view_as`, `view_as_complex`, `view_as_real`

## Indexing, masking, scatter, sorting, and set-like ops (30)

`argmax`, `argmin`, `argsort`, `argwhere`, `bucketize`, `diagonal`, `gather`, `index`, `index_add`, `index_copy`, `index_put`, `index_select`, `masked_fill`, `masked_scatter`, `masked_select`, `nonzero`, `put`, `scatter`, `scatter_add`, `scatter_reduce`, `searchsorted`, `sort`, `take_along_dim`, `topk`, `tril`, `triu`, `unique`, `unique_consecutive`, `values`, `where`

## Reductions, statistics, and histograms (18)

`all`, `amax`, `amin`, `any`, `bincount`, `cummax`, `cummin`, `cumsum`, `histc`, `logsumexp`, `max`, `mean`, `min`, `norm`, `prod`, `std`, `sum`, `var`

## Elementwise math, comparisons, and logical ops (54)

`abs`, `acos`, `add`, `asinh`, `bitwise_xor`, `ceil`, `clamp`, `clamp_max`, `clamp_min`, `clip`, `conj`, `cos`, `diff`, `div`, `eq`, `erf`, `exp`, `expm1`, `floor`, `floor_divide`, `fmod`, `ge`, `greater`, `isfinite`, `isin`, `isinf`, `isnan`, `log`, `log10`, `log1p`, `log2`, `logical_and`, `logical_not`, `logical_or`, `lt`, `maximum`, `minimum`, `mul`, `multiply`, `nan_to_num`, `ne`, `neg`, `pow`, `reciprocal`, `remainder`, `round`, `rsqrt`, `sign`, `sin`, `sinc`, `sqrt`, `square`, `sub`, `xlogy`

## Linear algebra, tensor contractions, and distances (9)

`baddbmm`, `bmm`, `cdist`, `cosine_similarity`, `cross`, `einsum`, `matmul`, `mm`, `outer`

## Activations and probability transforms (5)

`log_softmax`, `relu`, `sigmoid`, `softmax`, `tanh`

## Autograd, grad mode, and runtime controls (9)

`are_deterministic_algorithms_enabled`, `backward`, `detach`, `enable_grad`, `inference_mode`, `is_autocast_enabled`, `no_grad`, `register_hook`, `requires_grad`

## Predicates and introspection (3)

`is_contiguous`, `is_floating_point`, `is_tensor`

## Serialization (1)

`load`
