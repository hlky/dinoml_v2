# pe_audio representative config snapshot

Source basis:
- Transformers checkout: `transformers` at `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Hub access date: 2026-05-13
- Hub config URLs used:
  - https://huggingface.co/facebook/pe-a-frame-small/raw/main/config.json
  - https://huggingface.co/facebook/pe-a-frame-base/raw/main/config.json
  - https://huggingface.co/facebook/pe-a-frame-large/raw/main/config.json
  - https://huggingface.co/facebook/pe-a-frame-small/raw/main/preprocessor_config.json
  - https://huggingface.co/facebook/pe-a-frame-base/raw/main/preprocessor_config.json
  - https://huggingface.co/facebook/pe-a-frame-large/raw/main/preprocessor_config.json

Audio-only open checkpoints:

| Model id | Hub sha | architecture | audio hidden | audio layers | heads | head_dim | intermediate | text model |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `facebook/pe-a-frame-small` | not separately captured | `PeAudioFrameLevelModel` | 768 | 12 | 6 | 128 | 2048 | `modernbert` |
| `facebook/pe-a-frame-base` | `0a320953632f45113021e4bc8a0eda74f0024a74` | `PeAudioFrameLevelModel` | 1024 | 16 | 8 | 128 | 2752 | `modernbert` |
| `facebook/pe-a-frame-large` | `d7fd29e06effc0fe5630de3b6e4ca784876a1a16` | `PeAudioFrameLevelModel` | 1792 | 28 | 14 | 128 | 4800 | `modernbert` |

Shared open checkpoint fields:
- `audio_config.model_type`: `pe_audio_encoder`
- `audio_config.hidden_act`: `silu`
- `audio_config.num_key_value_heads == num_attention_heads`
- `audio_config.attention_bias`: `false`
- `audio_config.rms_norm_eps`: `1e-5`
- `audio_config.max_position_embeddings`: `10000`
- `audio_config.rope_parameters`: `{ "rope_theta": 20000, "rope_type": "default" }`
- `dac_config.model_type`: `dac`
- `dac_config.sampling_rate`: `48000`
- `dac_config.hop_length`: `1920`
- `dac_config.downsampling_ratios`: `[2, 8, 10, 12]`
- `dac_config.encoder_hidden_size`: `64`
- `dac_config.hidden_size`: `1024`
- `dac_config.codebook_dim`: `128`
- `dac_config.codebook_size`: `1024`
- `dac_config.n_codebooks`: `16`
- `text_config.model_type`: `modernbert`
- `text_config.hidden_size`: `1024`
- `text_config.num_hidden_layers`: `22`
- `text_config.num_attention_heads`: `16`
- `text_config.intermediate_size`: `2624`
- `text_config.vocab_size`: `50368`
- `text_config.max_position_embeddings`: `8192`
- `text_config.local_attention`: `128`
- `text_config.layer_types`: full attention every third layer, otherwise sliding attention

Shared preprocessor fields:
- `feature_extractor_type`: `PeAudioFeatureExtractor`
- `feature_size`: `1`
- `sampling_rate`: `48000`
- `hop_length`: `1920`
- `padding_side`: `right`
- `padding_value`: `0.0`
- `return_attention_mask`: `true`

Access notes:
- `https://huggingface.co/facebook/pe-a-base/raw/main/config.json` returned HTTP 401.
- `https://huggingface.co/facebook/pe-a-large/raw/main/config.json` returned HTTP 401.
- `facebook/pe-av-*` checkpoints use `model_type=pe_audio_video` and are not scoped as native `pe_audio`, though their nested audio branch shares the same audio encoder shape families.
