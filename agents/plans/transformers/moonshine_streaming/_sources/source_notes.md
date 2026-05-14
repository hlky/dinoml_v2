# moonshine_streaming source notes

## Transformers source basis

- Local source checkout: `X:/H/transformers`
- Inspected commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Generated runtime source:
  - `src/transformers/models/moonshine_streaming/modeling_moonshine_streaming.py`
  - `src/transformers/models/moonshine_streaming/configuration_moonshine_streaming.py`
  - `src/transformers/models/moonshine_streaming/processing_moonshine_streaming.py`
- Modular edit source:
  - `src/transformers/models/moonshine_streaming/modular_moonshine_streaming.py`
- Related inherited/reference source inspected:
  - `src/transformers/models/moonshine/modeling_moonshine.py`
  - `src/transformers/models/moonshine/configuration_moonshine.py`
  - `src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py`
  - `src/transformers/feature_extraction_sequence_utils.py`

Generated file banner says `modeling_moonshine_streaming.py` and
`processing_moonshine_streaming.py` are generated from
`modular_moonshine_streaming.py`; for runtime behavior this report cites the
generated files and notes the modular file as the future edit source.

## Hugging Face config snapshots

Fetched on 2026-05-13 from official, ungated UsefulSensors repos:

- [UsefulSensors/moonshine-streaming-tiny config](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny/raw/main/config.json), repo sha `f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a`
- [UsefulSensors/moonshine-streaming-small config](https://huggingface.co/UsefulSensors/moonshine-streaming-small/raw/main/config.json), repo sha `2c036506f23a09c18df5a50057599ba6d9280999`
- [UsefulSensors/moonshine-streaming-medium config](https://huggingface.co/UsefulSensors/moonshine-streaming-medium/raw/main/config.json), repo sha `57b843633a8c183cadf6699ffa761377a933a866`
- Shared processor/preprocessor shape:
  - `processor_class`: `MoonshineStreamingProcessor`
  - `feature_extractor_type`: `Wav2Vec2FeatureExtractor`
  - `sampling_rate`: `16000`
  - `do_normalize`: `false`
  - `return_attention_mask`: `true`
  - `pad_to_multiple_of`: `80`

Additional official ONNX bundle metadata was inspected only as streaming ABI
evidence, not as native Transformers graph behavior:

- [UsefulSensors/moonshine-streaming onnx/tiny/streaming_config.json](https://huggingface.co/UsefulSensors/moonshine-streaming/raw/main/onnx/tiny/streaming_config.json), repo sha `da30bae714913eb0057e733bccb4700104afb1b0`
- [UsefulSensors/moonshine-streaming onnx/small/streaming_config.json](https://huggingface.co/UsefulSensors/moonshine-streaming/raw/main/onnx/small/streaming_config.json)
- [UsefulSensors/moonshine-streaming onnx/medium/streaming_config.json](https://huggingface.co/UsefulSensors/moonshine-streaming/raw/main/onnx/medium/streaming_config.json)

The ONNX configs expose frontend state shapes:
`sample_buffer [1,79]`, `sample_len [1]`, `conv1_buffer [1, encoder_dim, 4]`,
`conv2_buffer [1, 2*encoder_dim, 4]`, and `frame_count [1]`, plus
`total_lookahead: 16`. The native Transformers source does not expose these
buffers in `forward`; it pads and processes whole `input_values`.

## Source anchors

- Encoder preprocessing:
  - frame length is `round(sample_rate * frame_ms / 1000.0)`, 80 samples for
    16 kHz / 5 ms.
  - waveform is reshaped to `[B, frames, frame_len]`, normalized by per-frame
    CMVN, passed through learned `asinh(exp(log_k) * x)`, then
    `Linear(frame_len -> encoder_hidden_size)` and SiLU.
  - two causal Conv1d layers use kernel 5, stride 2, left pad 4 each, with
    layout `[B, C, T]` between transposes.
- Encoder attention:
  - sliding-window masks are per layer from `encoder_config.sliding_windows`.
  - mask predicate admits `0 <= q_idx-kv_idx < left_window_size` or
    `0 < kv_idx-q_idx < right_window_size`.
- Decoder attention/cache:
  - `EncoderDecoderCache(DynamicCache, DynamicCache)` is created when
    `use_cache` is true.
  - self-attention applies RoPE before updating the self cache.
  - cross-attention caches projected encoder keys/values once and reuses them
    via `past_key_values.is_updated[layer_idx]`.
- RoPE:
  - default `rope_theta=10000`, partial rotary factor from config.
  - source repeats/interleaves cos/sin pair values and rotates only the prefix
    `rotary_dim`; the unrotated tail is concatenated back.
- Config gaps:
  - Native source reads `config.num_key_value_heads` for decoder attention, but
    `MoonshineStreamingConfig` source defaults do not declare it. Official
    checkpoints include it.
  - Official configs include `encoder_hidden_size` and `ffn_mult`; the inspected
    source does not read them directly.
  - Decoder query projection is reshaped with `num_key_value_heads`, so
    non-MHA decoder configs should be treated as unsafe unless the source is
    fixed or a checkpoint proves exact behavior.
