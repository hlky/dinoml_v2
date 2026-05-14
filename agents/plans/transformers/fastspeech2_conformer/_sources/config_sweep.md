# FastSpeech2 Conformer Config Sweep

Fetched from Hugging Face raw URLs on 2026-05-13. Values below are facts from `config.json` unless marked as source default.

## espnet/fastspeech2_conformer

URL: https://huggingface.co/espnet/fastspeech2_conformer/raw/main/config.json

```text
architectures = ["FastSpeech2ConformerModel"]
model_type = "fastspeech2_conformer"
torch_dtype = "float32"
transformers_version = "4.33.0.dev0"
hidden_size = 384
vocab_size = 78
input_dim = 78
num_mel_bins = 80
encoder_layers = 4
decoder_layers = 4
encoder_num_attention_heads = 2
decoder_config.num_attention_heads = 2
encoder_config.linear_units = 1536
decoder_config.linear_units = 1536
encoder_config.kernel_size = 7
decoder_config.kernel_size = 31
positionwise_conv_kernel_size = 3
use_macaron_style_in_conformer = true
use_cnn_in_conformer = true
duration_predictor_layers/channels/kernel/dropout = 2 / 256 / 3 / 0.2
pitch_predictor_layers/channels/kernel/dropout = 5 / 256 / 5 / 0.5
energy_predictor_layers/channels/kernel/dropout = 2 / 256 / 3 / 0.5
pitch_embed_kernel/dropout = 1 / 0.0
energy_embed_kernel/dropout = 1 / 0.0
reduction_factor = 1
speaking_speed = 1.0
num_speakers = null
num_languages = null
speaker_embed_dim = null
```

## espnet/fastspeech2_conformer_hifigan

URL: https://huggingface.co/espnet/fastspeech2_conformer_hifigan/raw/main/config.json

```text
architectures = ["FastSpeech2ConformerHifiGan"]
model_type = "hifigan"
torch_dtype = "float32"
transformers_version = "4.30.0.dev0"
model_in_dim = 80
sampling_rate = 22050
normalize_before = false
upsample_initial_channel = 512
upsample_rates = [8, 8, 2, 2]
upsample_kernel_sizes = [16, 16, 4, 4]
resblock_kernel_sizes = [3, 7, 11]
resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
leaky_relu_slope = 0.1
initializer_range = 0.01
```

Source gap: current `FastSpeech2ConformerHifiGanConfig.model_type` is `fastspeech2_conformer_hifigan`, not `hifigan`. Treat `hifigan` as a legacy public config value in this family context.

## espnet/fastspeech2_conformer_with_hifigan

URL: https://huggingface.co/espnet/fastspeech2_conformer_with_hifigan/raw/main/config.json

```text
architectures = ["FastSpeech2ConformerWithHifiGan"]
model_type = "fastspeech2_conformer_with_hifigan"
torch_dtype = "float32"
transformers_version = "4.33.0.dev0"
model_config.model_type = "fastspeech2_conformer"
vocoder_config.model_type = "hifigan"
```

Nested acoustic config matches `espnet/fastspeech2_conformer` for operator-significant fields. Nested vocoder config matches `espnet/fastspeech2_conformer_hifigan` for operator-significant fields.

## Tokenizer snapshot

URLs:

- https://huggingface.co/espnet/fastspeech2_conformer/raw/main/tokenizer_config.json
- https://huggingface.co/espnet/fastspeech2_conformer/raw/main/vocab.json

```text
tokenizer_class = "FastSpeech2ConformerTokenizer"
should_strip_spaces = true
pad_token = "<blank>" -> id 0
unk_token = "<unk>" -> id 1
bos_token = eos_token = "<sos/eos>" -> id 77
vocab size = 78
```

Tokenizer source dependency: `g2p_en` is required at tokenizer construction time.
