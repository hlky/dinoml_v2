# AIMv2 Config Snapshot

Source date: 2026-05-13.

Primary source checkout: `transformers` at `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Representative Hugging Face configs fetched from `https://huggingface.co/{model_id}/raw/main/config.json` and `preprocessor_config.json`.

| Model id | Repo sha from HF API | Architecture | Hidden | Layers | Heads | MLP | Image | Patches | Positional mode | Head | Params from HF API |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| `apple/aimv2-large-patch14-224` | `fcb5093a9c5b3efb7db5acee213849967fd18210` | `Aimv2VisionModel` | 1024 | 24 | 8 | 2816 | 224 | 256 | learned table | no | 309,197,824 |
| `apple/aimv2-huge-patch14-224` | `172051d9df825cda55607cfe9016da30f249a133` | `Aimv2VisionModel` | 1536 | 24 | 12 | 4096 | 224 | 256 | learned table | no | 680,851,968 |
| `apple/aimv2-1B-patch14-224` | `1f99ee5baffa9cfbebb0343b118458bb40c9c23c` | `Aimv2VisionModel` | 2048 | 24 | 16 | 5632 | 224 | 256 | learned table | no | 1,234,958,336 |
| `apple/aimv2-3B-patch14-224` | not separately fetched from API; config fetched | `Aimv2VisionModel` | 3072 | 24 | 24 | 8192 | 224 | 256 | learned table | no | collection says 2,720,658,432 |
| `apple/aimv2-large-patch14-336` | collection item fetched | `Aimv2VisionModel` | 1024 | 24 | 8 | 2816 | 336 | 576 | learned table | no | 309,525,504 |
| `apple/aimv2-large-patch14-448` | collection item fetched | `Aimv2VisionModel` | 1024 | 24 | 8 | 2816 | 448 | 1024 | learned table | no | 309,984,256 |
| `apple/aimv2-large-patch14-native` | `f733a03728c34470cf6df09bb16dc848b73e41ea` | `Aimv2VisionModel` | 1024 | 24 | 8 | 2816 | config says 224 | dynamic by input | generated 2D sinusoid | no | 308,935,680 |
| `apple/aimv2-large-patch14-224-lit` | `b17c109df4f9dbb941074073ad3771c28df5c826` | `Aimv2Model` | vision 1024, text 768 | vision 24, text 12 | vision 8, text 6 | vision 2816, text 2048 | 224 | 256 | learned vision + learned text | vision attention-pool | 436,680,192 |

Processor facts:

- Vision-only fixed-resolution repos use `CLIPImageProcessorFast`, `data_format="channels_first"`, RGB conversion, resize, center crop, rescale by `1/255`, CLIP mean/std, and output `pixel_values` as NCHW.
- Native repo disables resize and center crop; it still outputs channels-first normalized tensors and relies on input height/width divisible by patch size for exact patch-grid/position alignment.
- The `lit` repo uses the same image preprocessor plus `CLIPTokenizer` / `CLIPProcessor` metadata. Tokenizer max length is 77; BOS token id is 49406 and EOS/PAD/UNK are 49407.

Access gaps:

- `apple/aimv2-large-patch14-384` and `apple/aimv2-large-patch14-distilled` returned 401 for raw config URLs. The public collection lists distilled IDs as `apple/aimv2-large-patch14-224-distilled` and `apple/aimv2-large-patch14-336-distilled`; the mistyped `...-distilled` probe is not a valid representative source.
