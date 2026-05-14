# FastSpeech2 Conformer Source Notes

Inspection date: 2026-05-13.

Pinned source checkout:

```text
X:/H/transformers
git rev-parse HEAD = b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
```

Files inspected:

- `src/transformers/models/fastspeech2_conformer/modeling_fastspeech2_conformer.py`
- `src/transformers/models/fastspeech2_conformer/configuration_fastspeech2_conformer.py`
- `src/transformers/models/fastspeech2_conformer/tokenization_fastspeech2_conformer.py`
- `src/transformers/models/fastspeech2_conformer/convert_fastspeech2_conformer_original_pytorch_checkpoint_to_pytorch.py`
- `src/transformers/models/fastspeech2_conformer/convert_hifigan.py`
- `src/transformers/models/fastspeech2_conformer/convert_model_with_hifigan.py`
- `tests/models/fastspeech2_conformer/test_modeling_fastspeech2_conformer.py`

High-signal line references from the pinned checkout:

| Area | Local source lines |
| --- | --- |
| Length regulator | `modeling_fastspeech2_conformer.py:90-134` |
| Duration predictor | `modeling_fastspeech2_conformer.py:137-193` |
| Predictor/postnet Conv1d blocks | `modeling_fastspeech2_conformer.py:197-360` |
| Relative self-attention | `modeling_fastspeech2_conformer.py:362-470` |
| Conformer convolution module | `modeling_fastspeech2_conformer.py:474-543` |
| Conformer layer order | `modeling_fastspeech2_conformer.py:546-660` |
| Conv1d FFN replacement | `modeling_fastspeech2_conformer.py:663-706` |
| Relative positional encoding | `modeling_fastspeech2_conformer.py:709-777` |
| Encoder/decoder shared module | `modeling_fastspeech2_conformer.py:780-877` |
| Acoustic model init/forward | `modeling_fastspeech2_conformer.py:1037-1304` |
| HiFi-GAN residual/vocoder | `modeling_fastspeech2_conformer.py:1308-1491` |
| With-vocoder wrapper | `modeling_fastspeech2_conformer.py:1499-1604` |
| Acoustic config defaults/validation | `configuration_fastspeech2_conformer.py:145-262` |
| HiFi-GAN config defaults | `configuration_fastspeech2_conformer.py:265-323` |
| Wrapper config | `configuration_fastspeech2_conformer.py:326-380` |
| Tokenizer text cleanup/g2p/vocab behavior | `tokenization_fastspeech2_conformer.py:30-187` |
| Upstream mel integration shape | `test_modeling_fastspeech2_conformer.py:351-397` |
| Upstream training-label note | `test_modeling_fastspeech2_conformer.py:399-430` |
| Upstream wrapper waveform shape | `test_modeling_fastspeech2_conformer.py:770-790` |

Representative HF raw config URLs inspected:

- https://huggingface.co/espnet/fastspeech2_conformer/raw/main/config.json
- https://huggingface.co/espnet/fastspeech2_conformer_hifigan/raw/main/config.json
- https://huggingface.co/espnet/fastspeech2_conformer_with_hifigan/raw/main/config.json
- https://huggingface.co/espnet/fastspeech2_conformer/raw/main/tokenizer_config.json
- https://huggingface.co/espnet/fastspeech2_conformer/raw/main/vocab.json

Out-of-scope checkpoint probe:

- HF API search for ESPnet models matching `fastspeech2_conformer` returned many `library_name="espnet"` repos such as `espnet/kan-bayashi_ljspeech_conformer_fastspeech2`.
- Direct `config.json` fetch for representative ESPnet-library repos returned 404, so they are not Transformers-native `FastSpeech2ConformerConfig` checkpoints for this audit.

Important config/source gaps:

- Public vocoder configs use raw `"model_type": "hifigan"`, while the current source class declares `FastSpeech2ConformerHifiGanConfig.model_type = "fastspeech2_conformer_hifigan"`.
- Public acoustic config includes `input_dim=78`; current source uses `vocab_size`, not `input_dim`, in model construction.
- Public vocoder configs include `sampling_rate=22050`; current vocoder forward does not read sampling rate.
- The config class has strict validation for odd kernels and head divisibility; DinoML should mirror these as admission checks.
- No source `processing_*.py` or `feature_extraction_*.py` exists for this family.
