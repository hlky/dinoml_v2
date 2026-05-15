# Autoformer source hotspots

Transformers source basis: `transformers`, commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Inspected files:

- `src/transformers/models/autoformer/configuration_autoformer.py`
- `src/transformers/models/autoformer/modeling_autoformer.py`
- `src/transformers/time_series_utils.py`
- `tests/models/autoformer/test_modeling_autoformer.py`
- `docs/source/en/model_doc/autoformer.md`

Key implementation anchors:

- `AutoformerSeriesDecompositionLayer`, modeling lines 359-384: repeats first/last timestep by `(moving_average - 1) // 2`, applies `AvgPool1d(kernel_size=moving_average, stride=1)` over time after `B,T,C -> B,C,T`, returns `(x - trend, trend)`.
- `AutoformerLayernorm`, modeling lines 388-401: `LayerNorm(d_model)` followed by subtracting the time-axis mean of the normalized sequence.
- `AutoformerAttention`, modeling lines 404-604: Q/K/V linear projections, FFT autocorrelation via `rfft(q) * conj(rfft(k))`, `irfft`, top-k delay selection, softmax over selected delays, and roll/gather value aggregation. Test suite skips batching equivalence because `tmp_delay` logic is suspected wrong and `topk` is unstable.
- `AutoformerEncoderLayer`, modeling lines 606-664: AutoCorrelation self-attention, residual, `LayerNorm`, decomposition, FFN, decomposition, custom AutoformerLayernorm.
- `AutoformerDecoderLayer`, modeling lines 666-786: AutoCorrelation self-attention, optional cross AutoCorrelation, three decomposition layers, circular-padded `Conv1d(d_model -> feature_size, kernel=3)` trend projection, and final seasonality/trend split.
- `AutoformerModel.create_network_inputs`, modeling lines 1105-1204: scales context, builds static features from `log1p(abs(loc))` and `log(scale)`, appends static categorical/real and time features, gathers lagged subsequences, flattens lag dimension into features.
- `AutoformerForPrediction.generate`, modeling lines 1665-1830: repeats batch by `num_parallel_samples`, encodes once, decodes full horizon once, projects distribution params, samples output distribution, reshapes to `[B, samples, prediction_length, ...]`.
- `time_series_utils.py`, lines 65-225: distribution parameter projection and `StudentT`, `Normal`, `NegativeBinomial` domain maps.

