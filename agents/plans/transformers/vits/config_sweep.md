# VITS Config Sweep Snapshot

Fetched on 2026-05-13 from Hugging Face raw `config.json` and tokenizer files where accessible. Source basis is Transformers checkout `transformers` at `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

| Model id | Config access | Tokenizer access | Operator-significant fields |
| --- | --- | --- | --- |
| `facebook/mms-tts-eng` | Open raw config | Open raw tokenizer config and vocab | `vocab_size=38`, `num_speakers=1`, `speaker_embedding_size=0`, `sampling_rate=16000`, `hidden_size=192`, `num_hidden_layers=6`, `num_attention_heads=2`, stochastic duration predictor enabled. Tokenizer has `phonemize=false`, `normalize=true`, `add_blank=true`, `pad_token="k"`, language `eng`. |
| `facebook/mms-tts-spa` | Open raw config | Not fetched | `vocab_size=45`, `num_speakers=1`, `speaker_embedding_size=0`, `sampling_rate=16000`; otherwise same main topology as English MMS config. |
| `facebook/mms-tts-deu` | Opened via web raw config | Not fetched | `vocab_size=45`, `num_speakers=1`, `speaker_embedding_size=0`, `sampling_rate=16000`; same main topology. |
| `facebook/mms-tts-fra` | Opened via web raw config | Not fetched | `vocab_size=44`, `num_speakers=1`, `speaker_embedding_size=0`, `sampling_rate=16000`; same main topology. |
| `facebook/mms-tts-cmn` | HTTP 401 via raw config URL | Not fetched | Gap: repo/config was inaccessible without auth or because this id is not public under that name. A valid public Chinese MMS id or authorized access would resolve the language/tokenizer variation. |
| `kakao-enterprise/vits-ljs` | Open raw config | Not fetched | `vocab_size=178`, `num_speakers=1`, `speaker_embedding_size=0`, `sampling_rate=22050`; same hidden/flow/vocoder topology as source defaults. |
| `kakao-enterprise/vits-vctk` | Open raw config | Open raw tokenizer config and vocab | `vocab_size=178`, `num_speakers=109`, `speaker_embedding_size=256`, `sampling_rate=22050`; enables speaker embedding and conditioning conv paths in duration predictor, flow WaveNet, posterior encoder, and vocoder. Tokenizer has `phonemize=true`, `normalize=true`, `add_blank=true`, `pad_token="_"`, `language=null`. |

Common open-config values across representative checkpoints unless noted:

- `hidden_size=192`, `ffn_dim=768`, `flow_size=192`, `spectrogram_bins=513`.
- `num_hidden_layers=6`, `num_attention_heads=2`, effective `head_dim=96`, `window_size=4`, `use_bias=true`.
- `use_stochastic_duration_prediction=true`, `duration_predictor_num_flows=4`, `duration_predictor_flow_bins=10`, `duration_predictor_filter_channels=256`.
- `prior_encoder_num_flows=4`, `prior_encoder_num_wavenet_layers=4`, `posterior_encoder_num_wavenet_layers=16`.
- HiFi-GAN decoder: `upsample_initial_channel=512`, `upsample_rates=[8,8,2,2]`, `upsample_kernel_sizes=[16,16,4,4]`, `resblock_kernel_sizes=[3,7,11]`, `resblock_dilation_sizes=[[1,3,5],[1,3,5],[1,3,5]]`.
- Runtime stochastic controls: `speaking_rate=1.0`, `noise_scale=0.667`, `noise_scale_duration=0.8`.
