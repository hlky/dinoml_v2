# Forecasting ABI Excerpts

This snapshot records source-derived ABI details, paraphrased from the pinned
Transformers source.

## Config and feature width

Effective feature size:

```text
feature_size =
  input_size * len(lags_sequence)
  + sum(embedding_dimension)
  + num_dynamic_real_features
  + num_time_features
  + num_static_real_features
  + input_size * 2
```

The final `input_size * 2` terms are `log1p(abs(loc))` and `log(scale)`.
`context_length` defaults to `prediction_length` when omitted.

## Network input construction

`past_values` length must cover:

```text
past_length = context_length + max(lags_sequence)
```

`create_network_inputs`:

```text
time_feat =
  concat(past_time_features[:, past_length - context_length :], future_time_features)
    when future_values is provided
  otherwise past_time_features[:, past_length - context_length :]

context = past_values[:, -context_length:]
observed_context = past_observed_mask[:, -context_length:]
_, loc, scale = scaler(context, observed_context)

inputs =
  concat(past_values, future_values) normalized by loc/scale for teacher forcing
  otherwise past_values normalized by loc/scale

static_feat = concat(optional embedded static cats,
                    optional static real features,
                    log1p(abs(loc)),
                    log(scale))

features = concat(expand(static_feat over time), time_feat)
lagged_sequence = stack(sequence slices for each lag)
transformer_inputs = concat(flattened lagged_sequence, features)
```

Encoder consumes the first `context_length` rows. Decoder consumes remaining
rows in teacher-forcing mode. In forecast mode with no future values, the base
forward creates a single zero decoder input if there are no decoder rows.

## Lag extraction

For each effective lag `lag - shift`, source slices:

```text
sequence[:, -lag_index - subsequences_length : -lag_index]
```

with `end=None` when `lag_index == 0`. Generation calls this with `shift=1` and
`subsequences_length=1+k` at forecast step `k`.

## Scaling

Supported source scalers:

```text
scaling == "mean" or True:
  scale = mean(abs(data) over observed entries), with batch/default fallback.

scaling == "std":
  loc = observed mean
  scale = sqrt(observed variance + minimum_scale)

otherwise:
  loc = 0, scale = 1
```

Observed masks are float/bool tensors broadcast with `past_values`; missing
values are expected to be prefilled, usually with zero.

## Prediction heads

`distribution_output` selects one projection head:

```text
student_t:
  Linear(d_model -> input_size) for df
  Linear(d_model -> input_size) for loc
  Linear(d_model -> input_size) for scale
  df = 2 + squareplus(df)
  scale = clamp_min(squareplus(scale), dtype eps)

normal:
  Linear(d_model -> input_size) for loc
  Linear(d_model -> input_size) for scale
  scale = clamp_min(squareplus(scale), dtype eps)

negative_binomial:
  Linear(d_model -> input_size) for total_count
  Linear(d_model -> input_size) for logits
  total_count = squareplus(total_count)
  scale adjusts logits by adding log(scale)
```

For `input_size > 1`, distributions wrap the scalar distribution in
`torch.distributions.Independent(..., 1)`.

## Sampling loop

`TimeSeriesTransformerForPrediction.generate` first calls the model once with
`future_values=None` and `use_cache=True`, but then directly invokes the decoder
inside a Python loop:

```text
repeat batch by num_parallel_samples
repeat encoder last hidden state by num_parallel_samples
for k in 0..prediction_length-1:
  lagged_sequence = get_lagged_subsequences(repeated_past_values, 1+k, shift=1)
  decoder_input = concat(flatten(lagged_sequence), repeated_features[:, :k+1])
  dec_last_hidden = decoder(decoder_input, repeated_encoder_hidden).last_hidden_state
  params = parameter_projection(dec_last_hidden[:, -1:])
  next_sample = output_distribution(params, loc, scale).sample()
  append normalized next_sample to repeated_past_values
```

The source loop does not pass `past_key_values` to the decoder calls inside the
loop, so the practical source behavior recomputes the decoder prefix each step.
