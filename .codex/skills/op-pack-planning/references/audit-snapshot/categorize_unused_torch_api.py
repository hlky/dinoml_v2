from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path


API_KEYS = ("torch", "torch.Tensor", "torch.nn", "torch.nn.functional")


def load_analyzer(path: Path):
    spec = importlib.util.spec_from_file_location("torch_api_by_model_family", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load analyzer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def category_for_torch_tensor(name: str) -> str:
    if name in {
        "absolute",
        "acosh",
        "arccos",
        "arccosh",
        "arcsin",
        "arcsinh",
        "arctan",
        "arctan2",
        "arctanh",
        "asin",
        "atan",
        "atan2",
        "atanh",
        "cosh",
        "copysign",
        "deg2rad",
        "digamma",
        "erfc",
        "erfinv",
        "exp2",
        "fix",
        "frac",
        "gcd",
        "heaviside",
        "hypot",
        "i0",
        "igamma",
        "igammac",
        "isclose",
        "isneginf",
        "isposinf",
        "lcm",
        "ldexp",
        "lerp",
        "lgamma",
        "logaddexp",
        "logaddexp2",
        "logcumsumexp",
        "logit",
        "logical_xor",
        "mvlgamma",
        "nextafter",
        "polygamma",
        "positive",
        "rad2deg",
        "renorm",
        "sgn",
        "sinh",
        "tan",
        "trunc",
    } or name.startswith(("bitwise_", "special_")):
        return "Elementwise math, comparisons, and special functions"

    if any(
        part in name
        for part in (
            "batch_norm",
            "bilinear",
            "binary_cross_entropy",
            "celu",
            "channel_shuffle",
            "conv",
            "ctc",
            "dropout",
            "elu",
            "embedding",
            "fold",
            "gelu",
            "glu",
            "grid_sampler",
            "gru",
            "hardshrink",
            "hardsigmoid",
            "hardswish",
            "hardtanh",
            "huber",
            "instance_norm",
            "kl_div",
            "l1_loss",
            "layer_norm",
            "leaky_relu",
            "linear",
            "loss",
            "lstm",
            "margin",
            "mish",
            "nll",
            "pad",
            "pairwise_distance",
            "pixel_shuffle",
            "pixel_unshuffle",
            "prelu",
            "relu",
            "rnn",
            "selu",
            "softplus",
            "softshrink",
            "softsign",
            "tanhshrink",
            "threshold",
            "triplet",
        )
    ):
        return "Neural network ops, activations, losses, and fused kernels"

    if name.startswith(("fft", "istft", "stft")) or name.endswith("_window"):
        return "Signal processing, FFT, and window functions"

    if name in {
        "addbmm",
        "addmm",
        "addmv",
        "addr",
        "chain_matmul",
        "cholesky",
        "cholesky_inverse",
        "cholesky_solve",
        "det",
        "dot",
        "eig",
        "geqrf",
        "ger",
        "inner",
        "inverse",
        "kron",
        "lobpcg",
        "logdet",
        "lstsq",
        "lu",
        "lu_solve",
        "lu_unpack",
        "matrix_exp",
        "matrix_power",
        "matrix_rank",
        "matrix_transpose",
        "matrix_H",
        "mv",
        "orgqr",
        "ormqr",
        "pca_lowrank",
        "pinverse",
        "qr",
        "saddmm",
        "solve",
        "svd",
        "vdot",
        "vecdot",
    } or name.startswith(("linalg_", "_linalg")):
        return "Linear algebra and matrix decompositions"

    if name in {
        "aminmax",
        "corrcoef",
        "count_nonzero",
        "cov",
        "cumprod",
        "cumulative_trapezoid",
        "dist",
        "frexp",
        "gradient",
        "histogram",
        "histogramdd",
        "kthvalue",
        "median",
        "mode",
        "nanmean",
        "nanmedian",
        "nanquantile",
        "nansum",
        "quantile",
        "segment_reduce",
        "std_mean",
        "trapz",
        "trapezoid",
        "var_mean",
    }:
        return "Reductions, statistics, and numerical analysis"

    if name in {
        "binomial",
        "cauchy",
        "fbgemm_linear_fp16_weight",
        "fbgemm_linear_fp16_weight_fp32_activation",
        "fbgemm_linear_int8_weight",
        "fbgemm_linear_int8_weight_fp32_activation",
        "fbgemm_linear_quantize_weight",
        "fbgemm_pack_gemm_matrix_fp16",
        "fbgemm_pack_quantized_matrix",
        "fbgemm_pack_quantized_matrix_krow",
        "fbgemm_pack_quantized_matrix_krow_cpu",
        "fbgemm_pack_quantized_matrix_krow_cpu_tensor",
        "fbgemm_pack_quantized_matrix_krow_tensor",
        "fbgemm_pack_quantized_matrix_tensor",
        "fused_moving_avg_obs_fake_quant",
        "geometric",
        "log_normal",
        "poisson",
        "q_per_channel_axis",
        "q_per_channel_scales",
        "q_per_channel_zero_points",
        "q_scale",
        "q_zero_point",
        "qscheme",
        "quantize_per_channel",
        "quantize_per_tensor",
        "quantize_per_tensor_dynamic",
    } or "quant" in name:
        return "Random distributions and quantization"

    if any(
        part in name
        for part in (
            "bsc",
            "bsr",
            "coalesce",
            "col_indices",
            "csc",
            "csr",
            "dense",
            "indices",
            "jagged",
            "mkldnn",
            "nested",
            "sparse",
            "to_dense",
        )
    ):
        return "Sparse, compressed, nested, and layout-specific tensors"

    if name in {
        "affine_grid_generator",
        "align_as",
        "align_tensors",
        "align_to",
        "as_strided_copy",
        "as_strided_scatter",
        "asarray",
        "atleast_1d",
        "atleast_2d",
        "atleast_3d",
        "block_diag",
        "broadcast_shapes",
        "cartesian_prod",
        "column_stack",
        "combinations",
        "diag",
        "diag_embed",
        "diagflat",
        "dsplit",
        "dstack",
        "empty_permuted",
        "empty_strided",
        "from_dlpack",
        "frombuffer",
        "hsplit",
        "index_fill",
        "index_reduce",
        "is_same_size",
        "moveaxis",
        "msort",
        "new_empty",
        "new_empty_strided",
        "nonzero_static",
        "range",
        "ravel",
        "refine_names",
        "rename",
        "resize",
        "resize_as",
        "rot90",
        "row_stack",
        "select",
        "set",
        "slice_copy",
        "swapaxes",
        "swapdims",
        "take",
        "trace",
        "vsplit",
        "vstack",
    } or name.endswith(("_copy", "_scatter")):
        return "Tensor construction, shape, views, indexing, and copies"

    if any(
        part in name
        for part in ("cuda", "cudnn", "hip", "ipu", "mps", "mtia", "stream", "xpu")
    ) or name in {
        "autocast_decrement_nesting",
        "autocast_increment_nesting",
        "clear_autocast_cache",
        "compiled_with_cxx11_abi",
        "get_autocast_cpu_dtype",
        "get_autocast_gpu_dtype",
        "get_autocast_ipu_dtype",
        "get_autocast_xla_dtype",
        "get_default_device",
        "get_deterministic_debug_mode",
        "get_device",
        "get_device_module",
        "get_file_path",
        "get_float32_matmul_precision",
        "get_num_interop_threads",
        "get_num_threads",
        "get_rng_state",
        "init_num_threads",
        "initial_seed",
        "is_anomaly_enabled",
        "is_grad_enabled",
        "is_inference",
        "is_inference_mode_enabled",
        "is_warn_always_enabled",
        "is_vulkan_available",
        "pin_memory",
        "prepare_multiprocessing_environment",
        "random",
        "read_vitals",
        "set_anomaly_enabled",
        "set_autocast_cache_enabled",
        "set_autocast_cpu_dtype",
        "set_autocast_dtype",
        "set_autocast_enabled",
        "set_autocast_gpu_dtype",
        "set_autocast_ipu_dtype",
        "set_autocast_ipu_enabled",
        "set_autocast_xla_dtype",
        "set_autocast_xla_enabled",
        "set_default_device",
        "set_default_dtype",
        "set_deterministic_debug_mode",
        "set_float32_matmul_precision",
        "set_flush_denormal",
        "set_grad_enabled",
        "set_num_interop_threads",
        "set_num_threads",
        "set_printoptions",
        "set_rng_state",
        "set_vital",
        "use_deterministic_algorithms",
        "vitals_enabled",
    }:
        return "Devices, dtypes, autocast, RNG, and runtime configuration"

    if name in {
        "byte",
        "can_cast",
        "cdouble",
        "cfloat",
        "chalf",
        "char",
        "complex",
        "conj_physical",
        "double",
        "element_size",
        "float_power",
        "half",
        "imag",
        "int_repr",
        "is_complex",
        "is_conj",
        "is_floating_point",
        "is_nonzero",
        "is_signed",
        "isreal",
        "negative",
        "promote_types",
        "real",
        "resolve_conj",
        "resolve_neg",
        "result_type",
        "short",
    }:
        return "Dtype conversion and tensor property checks"

    if name in {
        "compile",
        "cond",
        "export",
        "fork",
        "func",
        "import_ir_module",
        "import_ir_module_from_buffer",
        "is_compiling",
        "jit",
        "library",
        "map",
        "map2",
        "merge_type_from_type_comment",
        "module_load",
        "parse_ir",
        "parse_schema",
        "parse_type_comment",
        "script",
        "script_if_tracing",
        "sym_constrain_range",
        "sym_constrain_range_for_size",
        "sym_float",
        "sym_int",
        "sym_ite",
        "sym_max",
        "sym_min",
        "sym_not",
        "vmap",
        "while_loop",
    }:
        return "Compilation, graph capture, control flow, and symbolic shapes"

    if name in {
        "data_ptr",
        "from_file",
        "hash_tensor",
        "is_shared",
        "save",
        "share_memory",
        "storage",
        "untyped_storage",
    }:
        return "Serialization, storage, and memory sharing"

    if name in {
        "apply",
        "as_subclass",
        "classproperty",
        "equal",
        "greater_equal",
        "gt",
        "handle_torch_function",
        "has_torch_function",
        "has_torch_function_unary",
        "has_torch_function_variadic",
        "implements",
        "is_storage",
        "le",
        "less",
        "less_equal",
        "nelement",
        "not_equal",
        "overrides",
        "register_post_accumulate_grad_hook",
        "reinforce",
        "requires_grad",
        "retain_grad",
        "return_types",
        "rsub",
        "typename",
    } or name.startswith(("has_", "is_")):
        return "Dispatch, overrides, predicates, and introspection"

    return "Miscellaneous low-level helpers and aliases"


def category_for_nn_module(name: str) -> str:
    if any(part in name for part in ("Conv", "Pool", "Unpool", "Pad", "Fold", "Pixel", "Shuffle", "Upsampling", "Unflatten")):
        return "Convolution, pooling, padding, and spatial reshaping modules"
    if any(part in name for part in ("Norm", "LRN")):
        return "Normalization modules"
    if "Dropout" in name:
        return "Dropout modules"
    if any(part in name for part in ("Loss", "Margin", "NLL", "CTC", "Huber")):
        return "Loss modules"
    if name in {
        "CELU",
        "Hardshrink",
        "Hardtanh",
        "LogSigmoid",
        "Mish",
        "PReLU",
        "RReLU",
        "ReLU6",
        "SELU",
        "Softmax2d",
        "Softmin",
        "Softshrink",
        "Softsign",
        "Tanhshrink",
        "Threshold",
    }:
        return "Activation modules"
    if any(part in name for part in ("RNN", "LSTM", "GRU")) or name.startswith("Transformer"):
        return "Sequence and Transformer modules"
    if name.startswith("Lazy"):
        return "Lazy initialization modules"
    if name in {"Bilinear", "CosineSimilarity", "EmbeddingBag", "PairwiseDistance"}:
        return "Embedding, linear, and distance modules"
    if name in {"Container", "DataParallel"}:
        return "Containers and parallel wrappers"
    return "Other nn modules"


def category_for_functional(name: str) -> str:
    if "pool" in name or "unpool" in name:
        return "Pooling and unpooling functions"
    if "conv" in name or name in {"bilinear", "grouped_mm", "scaled_grouped_mm", "scaled_mm"}:
        return "Convolution, bilinear, and matrix kernels"
    if "dropout" in name:
        return "Dropout functions"
    if any(part in name for part in ("norm", "normalize")) and "nll" not in name:
        return "Normalization functions"
    if any(part in name for part in ("loss", "cross_entropy", "ctc", "kl_div", "margin")):
        return "Loss functions"
    if name in {
        "celu",
        "elu",
        "gumbel_softmax",
        "hardshrink",
        "hardsigmoid",
        "hardswish",
        "hardtanh",
        "leaky_relu",
        "log_softmax",
        "mish",
        "prelu",
        "relu6",
        "rrelu",
        "selu",
        "softmin",
        "softshrink",
        "softsign",
        "tanh",
        "tanhshrink",
        "threshold",
    }:
        return "Activation and probability transform functions"
    if name in {"embedding_bag", "fold"}:
        return "Embedding and folding functions"
    if name in {"cosine_similarity", "pairwise_distance", "pdist"}:
        return "Distance and similarity functions"
    if name in {"affine_grid", "channel_shuffle", "native_channel_shuffle", "pixel_shuffle", "pixel_unshuffle"} or name.startswith("upsample"):
        return "Spatial transform, channel, and pixel layout functions"
    if name in {
        "Optional",
        "assert_int_or_pair",
        "boolean_dispatch",
        "handle_torch_function",
        "has_torch_function",
        "has_torch_function_unary",
        "has_torch_function_variadic",
    }:
        return "Dispatch and helper functions"
    return "Other functional APIs"


def group(names: list[str], categorizer) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for name in names:
        grouped[categorizer(name)].append(name)
    return dict(sorted(grouped.items()))


def emit_grouped(lines: list[str], title: str, grouped: dict[str, list[str]]) -> None:
    total = sum(len(names) for names in grouped.values())
    lines.extend([f"## {title} ({total})", ""])
    for category, names in grouped.items():
        lines.extend([f"### {category} ({len(names)})", ""])
        lines.append(", ".join(f"`{name}`" for name in names))
        lines.append("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Categorize unused public PyTorch APIs.")
    parser.add_argument(
        "transformers_dir",
        nargs="?",
        default=r"X:\H\transformers",
        help="Transformers repo root, src/transformers, or transformers package directory.",
    )
    parser.add_argument("--output", type=Path, default=Path("torch_api_unused_categories.md"))
    args = parser.parse_args()

    analyzer = load_analyzer(Path(__file__).with_name("torch_api_by_model_family.py"))
    public_api = analyzer.load_public_api()
    package = analyzer.resolve_transformers_package(args.transformers_dir)
    data = analyzer.scan(package, public_api)
    public_function_api = analyzer.load_public_function_api()

    torch_tensor = sorted(
        analyzer.public_names(public_function_api, ("torch", "torch.Tensor"), normalize_inplace=True)
        - set(analyzer.aggregate_counts(data, ("torch", "torch.Tensor"), normalize_inplace=True))
    )
    nn_modules = sorted(
        analyzer.public_nn_modules()
        - set(analyzer.aggregate_counts(data, ("torch.nn",)))
    )
    functional = sorted(
        analyzer.public_names(public_function_api, ("torch.nn.functional",), normalize_inplace=True)
        - set(analyzer.aggregate_counts(data, ("torch.nn.functional",), normalize_inplace=True))
    )

    lines = [
        "# Categorized Unused Public Torch APIs",
        "",
        f"Source: `{package}`",
        "",
        "Unused means absent from torch `modeling_*.py` files in the scanned Transformers tree. "
        "Function names are normalized so in-place suffixes such as `_` are folded into the base name.",
        "",
    ]
    emit_grouped(lines, "Unused torch and torch.Tensor functions", group(torch_tensor, category_for_torch_tensor))
    emit_grouped(lines, "Unused torch.nn modules", group(nn_modules, category_for_nn_module))
    emit_grouped(lines, "Unused torch.nn.functional functions", group(functional, category_for_functional))

    args.output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
