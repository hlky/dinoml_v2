# Kyutai Speech-to-Text Source Notes

Audit target: `kyutai_speech_to_text` only.

## Local source basis

- Transformers source repository: `X:/H/transformers`
- Required commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Verified local HEAD with `git -C X:/H/transformers rev-parse HEAD`.
- DinoML repository was not modified outside `H:/dinoml_v2/agents/plans/transformers/kyutai_speech_to_text/`.
- No imports, tests, or model execution were run.

## Files inspected

- `src/transformers/models/kyutai_speech_to_text/configuration_kyutai_speech_to_text.py`
- `src/transformers/models/kyutai_speech_to_text/modeling_kyutai_speech_to_text.py`
- `src/transformers/models/kyutai_speech_to_text/modular_kyutai_speech_to_text.py`
- `src/transformers/models/kyutai_speech_to_text/feature_extraction_kyutai_speech_to_text.py`
- `src/transformers/models/kyutai_speech_to_text/processing_kyutai_speech_to_text.py`
- `src/transformers/models/kyutai_speech_to_text/convert_kyutai_speech_to_text_to_hf.py`
- Neighbor/composed source:
  - `src/transformers/models/mimi/configuration_mimi.py`
  - `src/transformers/models/mimi/modeling_mimi.py`
  - `src/transformers/models/moshi/configuration_moshi.py`
  - `src/transformers/models/moshi/modeling_moshi.py`

## Representative configs and metadata

### `kyutai/stt-2.6b-en-trfs`

- HF model URL: <https://huggingface.co/kyutai/stt-2.6b-en-trfs>
- HF API SHA observed: `005de8e7800698a4c9963a5ac000e185b410c2f5`
- Gated: `false`
- License metadata: `cc-by-4.0`
- Library metadata: `transformers`
- Safetensors metadata: `BF16=2466449408`, `F32=246824801`, total `2696431425` parameters.
- `config.json` highlights:
  - `model_type="kyutai_speech_to_text"`
  - `torch_dtype="bfloat16"`
  - `vocab_size=4001`, `codebook_vocab_size=2049`
  - `hidden_size=2048`, `num_hidden_layers=48`
  - `num_attention_heads=32`, `num_key_value_heads=32`, `head_dim=64`
  - `ffn_dim=11264`, `hidden_act="silu"`
  - `max_position_embeddings=750`, `sliding_window=375`, `rope_theta=100000.0`
  - `num_codebooks=32`, `audio_bos_token_id=2048`, `audio_pad_token_id=69569`
  - `frame_size=1920`
  - nested Mimi codec: `sampling_rate=24000`, `audio_channels=1`, `hidden_size=512`, `num_hidden_layers=8`, `num_attention_heads=8`, `num_key_value_heads=8`, `head_dim=64`, `sliding_window=250`, `num_quantizers=32`, `codebook_size=2048`, `frame_size=1920` by source property, `use_streaming=false`.
- `preprocessor_config.json` highlights:
  - `sampling_rate=24000`, `feature_size=1`, `padding_side="right"`, `padding_value=0.0`
  - `audio_delay_seconds=2.5`, `audio_silence_prefix_seconds=1.0`
  - `chunk_length_s=null`, `overlap=null`, `return_attention_mask=true`
- `generation_config.json` highlights:
  - `audio_window_size=1`
  - `cache_implementation="sliding_window"`
  - `codec_cache_implementation="sliding_window"`
  - `codec_use_cache=true`
  - `bos_token_id=48000`, `pad_token_id=3`
- `tokenizer_config.json` highlights:
  - `tokenizer_class="PreTrainedTokenizerFast"`
  - `processor_class="KyutaiSpeechToTextProcessor"`
  - special tokens include `<unk>=0`, `<s>=1`, `</s>=2`, `<pad>=3`
  - tokenizer config leaves `bos_token_id/eos_token_id/pad_token_id` as `null`, while model/generation config supplies BOS/PAD IDs.

### Kyutai external-runtime repos

- `kyutai/stt-2.6b-en`: <https://huggingface.co/kyutai/stt-2.6b-en>
  - `library_name="moshi"`, `model_type="stt"`, not Transformers-native.
  - `config.json`: `text_card=4000`, `num_heads=32`, `num_layers=48`, `context=375`, `max_period=100000.0`, `audio_delay_seconds=2.5`, `audio_silence_prefix_seconds=1.0`, `mimi_name="mimi-pytorch-e351c8d8@125.safetensors"`, `tokenizer_name="tokenizer_en_audio_4000.model"`.
- `kyutai/stt-1b-en_fr`: <https://huggingface.co/kyutai/stt-1b-en_fr>
  - `library_name="moshi"`, `model_type="stt"`, not Transformers-native.
  - `config.json`: `text_card=8000`, `num_heads=16`, `num_layers=16`, `context=750`, `max_period=100000.0`, `audio_delay_seconds=0.5`, `audio_silence_prefix_seconds=0.0`, English/French tokenizer.

## Web/docs inspected

- Transformers Kyutai Speech-To-Text docs: <https://huggingface.co/docs/transformers/model_doc/kyutai_speech_to_text>
- Kyutai STT project page: <https://kyutai.org/stt>
  - Notes streaming STT, delayed-streams modeling, semantic VAD availability in Rust server, delay values, external PyTorch/Rust/MLX implementations.

## Source-derived implementation notes

- `KyutaiSpeechToTextForConditionalGeneration` owns:
  - `model`: autoregressive decoder over packed text/audio-token streams.
  - `lm_head`: text logits only.
  - `codec_model`: `AutoModel.from_config(config.codec_config)`, expected to be Mimi.
- The decoder input id tensor is rank 3 during generation: `[batch, sequence, 1 + num_codebooks]`, where slot 0 is the text token and slots 1..N are Mimi audio codes.
- `KyutaiSpeechToTextEmbeddings` stores one shared table of shape `vocab_size + num_codebooks * codebook_vocab_size + 1`; non-pad audio tokens receive per-codebook offsets before lookup, then embeddings are summed across the stream axis.
- During generation, `input_values` is chunked into `audio_window_size * frame_size` samples, encoded by Mimi, transposed from `[batch, codebooks, frames]` to `[batch, frames, codebooks]`, copied into `audio_tokens`, and stitched beside the current text input ids.
- `generate()` caps `max_new_tokens` to `input_values.shape[-1] // codec_config.frame_size` and explicitly ignores `padding_mask` for this cap.
- Mimi encoding path uses causal Conv1d/SEANet encoder, Mimi transformer, downsample Conv1d, split residual vector quantizer, and returns discrete codes.
- Mimi streaming state includes both transformer KV cache and causal Conv1d padding cache; Kyutai initializes both in `_prepare_model_inputs`.
- Main decoder cache uses GenerationMixin cache implementation from `generation_config`, usually sliding window for the representative Transformers checkpoint.

## Gaps / unresolved questions

- The Transformers-native checkpoint exists for `stt-2.6b-en-trfs`; the 1B English/French and original 2.6B repos are `library_name="moshi"` and have different source/config schema.
- The Kyutai project page says semantic VAD is available in the Rust server but not yet in other implementations; the inspected Transformers source does not expose a VAD head.
- The processor/tokenizer config ID mismatch is real: tokenizer config has `bos/eos/pad` null, while model/generation config uses `bos_token_id=48000` and `pad_token_id=3`.
- The code path says `padding_mask` is not used in the generation max-token cap, matching the original codebase. Batched variable-length audio requires care.
