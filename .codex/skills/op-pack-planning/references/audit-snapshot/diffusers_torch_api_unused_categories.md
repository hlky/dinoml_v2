# Categorized Public Torch APIs Not Used by Diffusers Model Components

Source: `X:\H\diffusers\src\diffusers`

Unused means absent from the selected Diffusers model component files. Function names are normalized so in-place suffixes such as `_` are folded into the base name.

## torch and torch.Tensor functions (641)

### Compilation, graph capture, control flow, and symbolic shapes (22)

`compile`, `cond`, `fork`, `import_ir_module`, `import_ir_module_from_buffer`, `map`, `map2`, `merge_type_from_type_comment`, `module_load`, `parse_ir`, `parse_schema`, `parse_type_comment`, `sym_constrain_range`, `sym_constrain_range_for_size`, `sym_float`, `sym_int`, `sym_ite`, `sym_max`, `sym_min`, `sym_not`, `vmap`, `while_loop`

### Devices, dtypes, autocast, RNG, and runtime configuration (60)

`autocast_decrement_nesting`, `autocast_increment_nesting`, `clear_autocast_cache`, `compiled_with_cxx11_abi`, `cuda`, `cudnn_affine_grid_generator`, `cudnn_is_acceptable`, `flipud`, `get_autocast_cpu_dtype`, `get_autocast_gpu_dtype`, `get_autocast_ipu_dtype`, `get_autocast_xla_dtype`, `get_default_device`, `get_deterministic_debug_mode`, `get_device`, `get_device_module`, `get_file_path`, `get_float32_matmul_precision`, `get_num_interop_threads`, `get_num_threads`, `get_rng_state`, `init_num_threads`, `initial_seed`, `ipu`, `is_anomaly_enabled`, `is_autocast_ipu_enabled`, `is_inference`, `is_inference_mode_enabled`, `is_vulkan_available`, `is_warn_always_enabled`, `mtia`, `pin_memory`, `prepare_multiprocessing_environment`, `profiler_allow_cudagraph_cupti_lazy_reinit_cuda12`, `random`, `read_vitals`, `record_stream`, `set_anomaly_enabled`, `set_autocast_cache_enabled`, `set_autocast_cpu_dtype`, `set_autocast_dtype`, `set_autocast_enabled`, `set_autocast_gpu_dtype`, `set_autocast_ipu_dtype`, `set_autocast_ipu_enabled`, `set_autocast_xla_dtype`, `set_autocast_xla_enabled`, `set_default_device`, `set_default_dtype`, `set_deterministic_debug_mode`, `set_float32_matmul_precision`, `set_flush_denormal`, `set_num_interop_threads`, `set_num_threads`, `set_printoptions`, `set_rng_state`, `set_vital`, `use_deterministic_algorithms`, `vitals_enabled`, `xpu`

### Dispatch, overrides, predicates, and introspection (29)

`apply`, `as_subclass`, `classproperty`, `equal`, `greater_equal`, `gt`, `has_names`, `is_anomaly_check_nan_enabled`, `is_autocast_cache_enabled`, `is_autocast_cpu_enabled`, `is_autocast_enabled`, `is_autocast_xla_enabled`, `is_contiguous`, `is_deterministic_algorithms_warn_only_enabled`, `is_distributed`, `is_neg`, `is_pinned`, `is_set_to`, `is_storage`, `le`, `less`, `less_equal`, `nelement`, `not_equal`, `register_post_accumulate_grad_hook`, `reinforce`, `retain_grad`, `rsub`, `typename`

### Dtype conversion and tensor property checks (24)

`can_cast`, `cdouble`, `cfloat`, `chalf`, `char`, `complex`, `conj_physical`, `float_power`, `half`, `imag`, `int_repr`, `is_complex`, `is_conj`, `is_floating_point`, `is_nonzero`, `is_signed`, `isreal`, `negative`, `promote_types`, `real`, `resolve_conj`, `resolve_neg`, `result_type`, `short`

### Elementwise math, comparisons, and special functions (54)

`absolute`, `acosh`, `arccos`, `arccosh`, `arcsin`, `arcsinh`, `arctan`, `arctan2`, `arctanh`, `asin`, `atan`, `atan2`, `bitwise_and`, `bitwise_left_shift`, `bitwise_not`, `bitwise_or`, `bitwise_right_shift`, `bitwise_xor`, `copysign`, `cosh`, `deg2rad`, `digamma`, `erfc`, `erfinv`, `exp2`, `fix`, `frac`, `gcd`, `heaviside`, `hypot`, `i0`, `igamma`, `igammac`, `isclose`, `isneginf`, `isposinf`, `lcm`, `ldexp`, `lgamma`, `logaddexp`, `logaddexp2`, `logcumsumexp`, `logical_xor`, `logit`, `mvlgamma`, `nextafter`, `polygamma`, `positive`, `rad2deg`, `renorm`, `sgn`, `sinh`, `tan`, `trunc`

### Linear algebra and matrix decompositions (35)

`addbmm`, `addmm`, `addmv`, `addr`, `chain_matmul`, `cholesky`, `cholesky_inverse`, `cholesky_solve`, `det`, `dot`, `eig`, `geqrf`, `ger`, `inner`, `inverse`, `kron`, `lobpcg`, `logdet`, `lstsq`, `lu`, `lu_solve`, `lu_unpack`, `matrix_exp`, `matrix_power`, `matrix_rank`, `mv`, `orgqr`, `ormqr`, `pca_lowrank`, `pinverse`, `qr`, `saddmm`, `solve`, `svd`, `vdot`

### Miscellaneous low-level helpers and aliases (156)

`acos`, `adaptive_avg_pool1d`, `adaptive_max_pool1d`, `addcdiv`, `adjoint`, `allclose`, `amax`, `amin`, `angle`, `are_deterministic_algorithms_enabled`, `argwhere`, `as_strided`, `asinh`, `avg_pool1d`, `backward`, `bernoulli`, `bfloat16`, `broadcast_tensors`, `bucketize`, `ceil`, `choose_qparams_optimized`, `clamp_max`, `clamp_min`, `conj`, `cosine_similarity`, `cross`, `cummax`, `cummin`, `diagonal`, `diff`, `dim_order`, `divide`, `dsmm`, `eq`, `erf`, `expand_as`, `expm1`, `exponential`, `eye`, `fill`, `fill_diagonal`, `fliplr`, `floor`, `floor_divide`, `fmax`, `fmin`, `fmod`, `frobenius_norm`, `ge`, `get_autocast_dtype`, `get_default_dtype`, `greater`, `group_norm`, `histc`, `hsmm`, `hspmm`, `hstack`, `index`, `index_add`, `index_put`, `index_select`, `isfinite`, `isin`, `isinf`, `isnan`, `load`, `log10`, `log1p`, `log2`, `log_softmax`, `logical_and`, `logsumexp`, `lt`, `manual_seed`, `masked_select`, `max_pool1d`, `max_pool2d`, `max_pool3d`, `min`, `minimum`, `mm`, `multinomial`, `multiply`, `nan_to_num`, `narrow`, `native_group_norm`, `native_norm`, `ndimension`, `ne`, `neg`, `new`, `new_full`, `new_tensor`, `nonzero`, `norm_except_dim`, `normal`, `nuclear_norm`, `pdist`, `prod`, `put`, `rand_like`, `randint_like`, `randn_like`, `randperm`, `register_hook`, `remainder`, `reshape_as`, `rms_norm`, `roll`, `scalar_tensor`, `scatter`, `searchsorted`, `seed`, `set_autocast_cpu_enabled`, `set_default_tensor_type`, `set_warn_always`, `sign`, `signbit`, `sinc`, `slice_inverse`, `slogdet`, `smm`, `spmm`, `square`, `std`, `storage_offset`, `storage_type`, `stride`, `subtract`, `sum_to_size`, `svd_lowrank`, `sym_fresh_size`, `sym_sqrt`, `sym_sum`, `symeig`, `t`, `take_along_dim`, `tensor_split`, `tensordot`, `tile`, `to_dlpack`, `triangular_solve`, `true_divide`, `uniform`, `unify_type_list`, `unique`, `unique_consecutive`, `unravel_index`, `unsafe_chunk`, `unsafe_split`, `unsafe_split_with_sizes`, `vander`, `var`, `view_as`, `wait`, `xlogy`

### Neural network ops, activations, losses, and fused kernels (93)

`alpha_dropout`, `batch_norm`, `batch_norm_backward_elemt`, `batch_norm_backward_reduce`, `batch_norm_elemt`, `batch_norm_gather_stats`, `batch_norm_gather_stats_with_counts`, `batch_norm_stats`, `batch_norm_update_stats`, `bilinear`, `binary_cross_entropy_with_logits`, `celu`, `channel_shuffle`, `constant_pad_nd`, `conv1d`, `conv2d`, `conv3d`, `conv_tbc`, `conv_transpose1d`, `conv_transpose2d`, `conv_transpose3d`, `convolution`, `cosine_embedding_loss`, `ctc_loss`, `cudnn_batch_norm`, `cudnn_convolution`, `cudnn_convolution_add_relu`, `cudnn_convolution_relu`, `cudnn_convolution_transpose`, `cudnn_grid_sampler`, `dropout`, `embedding`, `embedding_bag`, `embedding_renorm`, `fbgemm_linear_fp16_weight`, `fbgemm_linear_fp16_weight_fp32_activation`, `fbgemm_linear_int8_weight`, `fbgemm_linear_int8_weight_fp32_activation`, `fbgemm_linear_quantize_weight`, `feature_alpha_dropout`, `feature_dropout`, `grid_sampler`, `grid_sampler_2d`, `grid_sampler_3d`, `gru`, `gru_cell`, `hardshrink`, `hinge_embedding_loss`, `instance_norm`, `kl_div`, `layer_norm`, `lstm`, `lstm_cell`, `margin_ranking_loss`, `miopen_batch_norm`, `miopen_convolution`, `miopen_convolution_add_relu`, `miopen_convolution_relu`, `miopen_convolution_transpose`, `miopen_ctc_loss`, `miopen_depthwise_convolution`, `miopen_rnn`, `mkldnn_convolution`, `mkldnn_linear_backward_weights`, `mkldnn_rnn_layer`, `native_batch_norm`, `native_channel_shuffle`, `native_dropout`, `native_layer_norm`, `pairwise_distance`, `pixel_shuffle`, `pixel_unshuffle`, `poisson_nll_loss`, `prelu`, `quantized_batch_norm`, `quantized_gru`, `quantized_gru_cell`, `quantized_lstm`, `quantized_lstm_cell`, `quantized_rnn_relu_cell`, `quantized_rnn_tanh_cell`, `rnn_relu`, `rnn_relu_cell`, `rnn_tanh`, `rnn_tanh_cell`, `rrelu`, `selu`, `sspaddmm`, `threshold`, `to_padded_tensor`, `triplet_margin_loss`, `unfold`, `unfold_copy`

### Random distributions and quantization (24)

`binomial`, `cauchy`, `dequantize`, `empty_quantized`, `fake_quantize_per_channel_affine`, `fake_quantize_per_tensor_affine`, `fbgemm_pack_gemm_matrix_fp16`, `fbgemm_pack_quantized_matrix`, `fused_moving_avg_obs_fake_quant`, `geometric`, `log_normal`, `poisson`, `q_per_channel_axis`, `q_per_channel_scales`, `q_per_channel_zero_points`, `q_scale`, `q_zero_point`, `qscheme`, `quantize_per_channel`, `quantize_per_tensor`, `quantize_per_tensor_dynamic`, `quantized_max_pool1d`, `quantized_max_pool2d`, `quantized_max_pool3d`

### Reductions, statistics, and numerical analysis (22)

`aminmax`, `corrcoef`, `count_nonzero`, `cov`, `cumulative_trapezoid`, `dist`, `frexp`, `gradient`, `histogram`, `histogramdd`, `kthvalue`, `median`, `nanmean`, `nanmedian`, `nanquantile`, `nansum`, `quantile`, `segment_reduce`, `std_mean`, `trapezoid`, `trapz`, `var_mean`

### Serialization, storage, and memory sharing (8)

`data_ptr`, `from_file`, `hash_tensor`, `is_shared`, `save`, `share_memory`, `storage`, `untyped_storage`

### Signal processing, FFT, and window functions (7)

`bartlett_window`, `blackman_window`, `hamming_window`, `hann_window`, `istft`, `kaiser_window`, `stft`

### Sparse, compressed, nested, and layout-specific tensors (38)

`ccol_indices`, `ccol_indices_copy`, `coalesce`, `col_indices`, `col_indices_copy`, `crow_indices`, `crow_indices_copy`, `dense_dim`, `indices`, `indices_copy`, `is_coalesced`, `max_pool1d_with_indices`, `mkldnn_adaptive_avg_pool2d`, `mkldnn_max_pool2d`, `mkldnn_max_pool3d`, `resize_as_sparse`, `row_indices`, `row_indices_copy`, `sparse_bsc_tensor`, `sparse_bsr_tensor`, `sparse_compressed_tensor`, `sparse_coo_tensor`, `sparse_csc_tensor`, `sparse_csr_tensor`, `sparse_dim`, `sparse_mask`, `sparse_resize`, `sparse_resize_and_clear`, `to_dense`, `to_mkldnn`, `to_sparse`, `to_sparse_bsc`, `to_sparse_bsr`, `to_sparse_coo`, `to_sparse_csc`, `to_sparse_csr`, `tril_indices`, `triu_indices`

### Tensor construction, shape, views, indexing, and copies (69)

`affine_grid_generator`, `alias_copy`, `align_as`, `align_tensors`, `align_to`, `as_strided_copy`, `as_strided_scatter`, `asarray`, `atleast_1d`, `atleast_2d`, `atleast_3d`, `block_diag`, `broadcast_shapes`, `cartesian_prod`, `column_stack`, `combinations`, `detach_copy`, `diag`, `diagflat`, `diagonal_copy`, `diagonal_scatter`, `dsplit`, `dstack`, `empty_permuted`, `empty_strided`, `expand_copy`, `from_dlpack`, `frombuffer`, `hsplit`, `index_copy`, `index_fill`, `index_reduce`, `is_same_size`, `masked_scatter`, `moveaxis`, `msort`, `narrow_copy`, `new_empty`, `new_empty_strided`, `nonzero_static`, `permute_copy`, `range`, `ravel`, `refine_names`, `rename`, `resize_as`, `rot90`, `row_stack`, `select`, `select_copy`, `select_scatter`, `set`, `slice_copy`, `slice_scatter`, `split_copy`, `split_with_sizes_copy`, `squeeze_copy`, `swapdims`, `t_copy`, `take`, `trace`, `transpose_copy`, `unbind_copy`, `unsqueeze_copy`, `values_copy`, `view_as_complex_copy`, `view_as_real_copy`, `view_copy`, `vsplit`

## torch.nn modules (130)

### Activation modules (15)

`CELU`, `Hardshrink`, `Hardtanh`, `LogSigmoid`, `Mish`, `PReLU`, `RReLU`, `ReLU6`, `SELU`, `Softmax2d`, `Softmin`, `Softshrink`, `Softsign`, `Tanhshrink`, `Threshold`

### Containers and parallel wrappers (2)

`Container`, `DataParallel`

### Convolution, pooling, padding, and spatial reshaping modules (42)

`AdaptiveAvgPool1d`, `AdaptiveAvgPool3d`, `AdaptiveMaxPool1d`, `AdaptiveMaxPool2d`, `AdaptiveMaxPool3d`, `ChannelShuffle`, `CircularPad1d`, `CircularPad2d`, `CircularPad3d`, `ConstantPad1d`, `ConstantPad2d`, `ConstantPad3d`, `ConvTranspose3d`, `Fold`, `FractionalMaxPool2d`, `FractionalMaxPool3d`, `LPPool1d`, `LPPool2d`, `LPPool3d`, `LazyConv1d`, `LazyConv2d`, `LazyConv3d`, `LazyConvTranspose1d`, `LazyConvTranspose2d`, `LazyConvTranspose3d`, `MaxPool1d`, `MaxPool2d`, `MaxPool3d`, `MaxUnpool1d`, `MaxUnpool2d`, `MaxUnpool3d`, `ReflectionPad1d`, `ReflectionPad2d`, `ReflectionPad3d`, `ReplicationPad1d`, `ReplicationPad2d`, `ReplicationPad3d`, `Unflatten`, `UpsamplingBilinear2d`, `UpsamplingNearest2d`, `ZeroPad1d`, `ZeroPad3d`

### Dropout modules (5)

`AlphaDropout`, `Dropout1d`, `Dropout2d`, `Dropout3d`, `FeatureAlphaDropout`

### Embedding, linear, and distance modules (4)

`Bilinear`, `CosineSimilarity`, `EmbeddingBag`, `PairwiseDistance`

### Lazy initialization modules (1)

`LazyLinear`

### Loss modules (23)

`AdaptiveLogSoftmaxWithLoss`, `BCELoss`, `BCEWithLogitsLoss`, `CTCLoss`, `CosineEmbeddingLoss`, `CrossEntropyLoss`, `GaussianNLLLoss`, `HingeEmbeddingLoss`, `HuberLoss`, `KLDivLoss`, `L1Loss`, `MSELoss`, `MarginRankingLoss`, `MultiLabelMarginLoss`, `MultiLabelSoftMarginLoss`, `MultiMarginLoss`, `NLLLoss`, `NLLLoss2d`, `PoissonNLLLoss`, `SmoothL1Loss`, `SoftMarginLoss`, `TripletMarginLoss`, `TripletMarginWithDistanceLoss`

### Normalization modules (14)

`BatchNorm1d`, `BatchNorm3d`, `CrossMapLRN2d`, `InstanceNorm1d`, `InstanceNorm2d`, `InstanceNorm3d`, `LazyBatchNorm1d`, `LazyBatchNorm2d`, `LazyBatchNorm3d`, `LazyInstanceNorm1d`, `LazyInstanceNorm2d`, `LazyInstanceNorm3d`, `LocalResponseNorm`, `SyncBatchNorm`

### Other nn modules (11)

`Flatten`, `GLU`, `Hardsigmoid`, `Hardswish`, `LogSoftmax`, `ParameterDict`, `ParameterList`, `Sigmoid`, `Softmax`, `Softplus`, `Unfold`

### Sequence and Transformer modules (13)

`GRU`, `GRUCell`, `LSTM`, `LSTMCell`, `RNN`, `RNNBase`, `RNNCell`, `RNNCellBase`, `Transformer`, `TransformerDecoder`, `TransformerDecoderLayer`, `TransformerEncoder`, `TransformerEncoderLayer`

## torch.nn.functional functions (100)

### Activation and probability transform functions (18)

`celu`, `elu`, `gumbel_softmax`, `hardshrink`, `hardsigmoid`, `hardswish`, `hardtanh`, `mish`, `prelu`, `relu6`, `rrelu`, `selu`, `softmin`, `softshrink`, `softsign`, `tanh`, `tanhshrink`, `threshold`

### Convolution, bilinear, and matrix kernels (4)

`bilinear`, `conv_tbc`, `scaled_grouped_mm`, `scaled_mm`

### Dispatch and helper functions (7)

`Optional`, `assert_int_or_pair`, `boolean_dispatch`, `handle_torch_function`, `has_torch_function`, `has_torch_function_unary`, `has_torch_function_variadic`

### Distance and similarity functions (3)

`cosine_similarity`, `pairwise_distance`, `pdist`

### Dropout functions (6)

`alpha_dropout`, `dropout`, `dropout1d`, `dropout2d`, `dropout3d`, `feature_alpha_dropout`

### Embedding and folding functions (2)

`embedding_bag`, `fold`

### Loss functions (21)

`binary_cross_entropy`, `binary_cross_entropy_with_logits`, `cosine_embedding_loss`, `cross_entropy`, `ctc_loss`, `gaussian_nll_loss`, `hinge_embedding_loss`, `huber_loss`, `kl_div`, `l1_loss`, `margin_ranking_loss`, `mse_loss`, `multi_margin_loss`, `multilabel_margin_loss`, `multilabel_soft_margin_loss`, `nll_loss`, `poisson_nll_loss`, `smooth_l1_loss`, `soft_margin_loss`, `triplet_margin_loss`, `triplet_margin_with_distance_loss`

### Normalization functions (4)

`batch_norm`, `instance_norm`, `local_response_norm`, `rms_norm`

### Other functional APIs (5)

`glu`, `grid_sample`, `logsigmoid`, `sigmoid`, `unfold`

### Pooling and unpooling functions (24)

`adaptive_avg_pool1d`, `adaptive_avg_pool2d`, `adaptive_avg_pool3d`, `adaptive_max_pool1d`, `adaptive_max_pool1d_with_indices`, `adaptive_max_pool2d`, `adaptive_max_pool2d_with_indices`, `adaptive_max_pool3d`, `adaptive_max_pool3d_with_indices`, `fractional_max_pool2d`, `fractional_max_pool2d_with_indices`, `fractional_max_pool3d`, `fractional_max_pool3d_with_indices`, `lp_pool1d`, `lp_pool2d`, `lp_pool3d`, `max_pool1d`, `max_pool1d_with_indices`, `max_pool2d`, `max_pool2d_with_indices`, `max_pool3d_with_indices`, `max_unpool1d`, `max_unpool2d`, `max_unpool3d`

### Spatial transform, channel, and pixel layout functions (6)

`affine_grid`, `channel_shuffle`, `native_channel_shuffle`, `upsample`, `upsample_bilinear`, `upsample_nearest`
