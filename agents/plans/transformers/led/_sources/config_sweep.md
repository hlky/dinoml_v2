# LED config sweep notes

Source basis: raw `config.json` files fetched from Hugging Face on 2026-05-13, plus local Transformers source snapshots copied from `X:/H/transformers/src/transformers/models/led`.

| Model id | Config snapshot | Architecture | d_model | Encoder layers | Decoder layers | Heads | FFN | Max encoder | Max decoder | Attention window | Vocab | Generation notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| `allenai/led-base-16384` | `allenai__led-base-16384.config.json` | `LEDForConditionalGeneration` | 768 | 6 | 6 | 12 | 3072 | 16384 | 1024 | 6 x 1024 | 50265 | generation config only has BOS/EOS/PAD/start ids |
| `allenai/led-large-16384` | `allenai__led-large-16384.config.json` | `LEDForConditionalGeneration` | 1024 | 12 | 12 | 16 | 4096 | 16384 | 1024 | 12 x 1024 | 50265 | generation config only has BOS/EOS/PAD/start ids |
| `allenai/led-large-16384-arxiv` | `allenai__led-large-16384-arxiv.config.json` | `LEDForConditionalGeneration` | 1024 | 12 | 12 | 16 | 4096 | 16384 | 1024 | 12 x 1024 | 50265 | `max_length=512`, `num_beams=4` |
| `patrickvonplaten/led-large-16384-pubmed` | `patrickvonplaten__led-large-16384-pubmed.config.json` | `LEDForConditionalGeneration` | 1024 | 12 | 12 | 16 | 4096 | 16384 | 1024 | 12 x 1024 | 50265 | `max_length=512`, `num_beams=4` |
| `HHousen/distil-led-large-cnn-16384` | `HHousen__distil-led-large-cnn-16384.config.json` | `LEDForConditionalGeneration` | 1024 | 12 | 6 | 16 | 4096 | 16384 | 1024 | 12 x 1024 | 50264 | `max_length=142`, `num_beams=4`; distilled decoder depth and vocab differ |
| `MingZhong/DialogLED-large-5120` | `MingZhong__DialogLED-large-5120.config.json` | `LEDForConditionalGeneration` | 1024 | 12 | 12 | 16 | 4096 | 16384 in config despite model name | 1024 | 12 x 1024 | 50265 | `max_length=512`, `num_beams=6` |
| `allenai/PRIMERA` | `allenai__PRIMERA.config.json` | `LEDForConditionalGeneration` | 1024 | 12 | 12 | 16 | 4096 | 4096 | 1024 | 12 x 512 | 50266 | variant changes encoder context/window and vocab |

Access note: `allenai/led-large-16384-pubmed` returned 401; `patrickvonplaten/led-large-16384-pubmed` was accessible and has the same LED source class.
