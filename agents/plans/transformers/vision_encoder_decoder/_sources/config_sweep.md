# vision_encoder_decoder representative config sweep

Fetched from public Hugging Face model repos on 2026-05-13 using raw `config.json`, `preprocessor_config.json`, and `tokenizer_config.json` URLs where present.

## Checkpoints

| Model id | Config URL | Processor/tokenizer notes | Wrapper architecture |
|---|---|---|---|
| `microsoft/trocr-base-handwritten` | https://huggingface.co/microsoft/trocr-base-handwritten/raw/main/config.json | `preprocessor_config.json`: `ViTImageProcessor`, resize 384, normalize mean/std `[0.5,0.5,0.5]`; tokenizer is `RobertaTokenizer`. No `processor_config.json` in repo. | ViT encoder hidden 768, TrOCR decoder hidden 1024, `cross_attention_hidden_size=768`; wrapper does not insert `enc_to_dec_proj`. |
| `microsoft/trocr-small-printed` | https://huggingface.co/microsoft/trocr-small-printed/raw/main/config.json | `preprocessor_config.json`: `DeiTImageProcessor`, resize 384, normalize `[0.5]*3`; tokenizer is `XLMRobertaTokenizer`. | DeiT encoder hidden 384, TrOCR decoder hidden 256, `cross_attention_hidden_size=384`; wrapper does not insert `enc_to_dec_proj`. |
| `nlpconnect/vit-gpt2-image-captioning` | https://huggingface.co/nlpconnect/vit-gpt2-image-captioning/raw/main/config.json | `preprocessor_config.json`: legacy `ViTFeatureExtractor`, resize 224, normalize `[0.5]*3`; tokenizer is `GPT2Tokenizer`. | ViT encoder hidden 768, GPT-2 decoder hidden 768, no projection needed. |
| `ydshieh/vit-gpt2-coco-en` | https://huggingface.co/ydshieh/vit-gpt2-coco-en/raw/main/config.json | `preprocessor_config.json`: legacy `ViTFeatureExtractor`, resize 224, normalize `[0.5]*3`; tokenizer is `GPT2Tokenizer`. | ViT encoder hidden 768, GPT-2 decoder hidden 768, no projection needed. |

## Config-derived dimensions

| Model id | Encoder | Encoder dims | Decoder | Decoder dims | Image | Decoder vocab/positions | Cache default |
|---|---|---:|---|---:|---:|---:|---|
| `microsoft/trocr-base-handwritten` | `vit` | H=768, L=12, A=12, patch=16, qkv_bias=false | `trocr` | H=1024, L=12, A=16, FFN=4096, activation=gelu | 384 -> 577 tokens | vocab 50265, max pos 512 | decoder config serializes `use_cache=false` |
| `microsoft/trocr-small-printed` | `deit` | H=384, L=12, A=6, patch=16, qkv_bias=true | `trocr` | H=256, L=6, A=8, FFN=1024, activation=relu | 384 -> 577 tokens | vocab 64044, max pos 512 | decoder config serializes `use_cache=false` |
| `nlpconnect/vit-gpt2-image-captioning` | `vit` | H=768, L=12, A=12, patch=16, qkv_bias=true | `gpt2` | H=768, L=12, A=12, FFN default 4H, activation=gelu_new | 224 -> 197 tokens | vocab 50257, max pos 1024 | `use_cache=true` |
| `ydshieh/vit-gpt2-coco-en` | `vit` | H=768, L=12, A=12, patch=16, qkv_bias=true | `gpt2` | H=768, L=12, A=12, FFN default 4H, activation=gelu_new | 224 -> 197 tokens | vocab 50257, max pos 1024 | `use_cache=true` |

## Processor ABI observations

- The wrapper's `main_input_name` is `pixel_values`.
- ViT/DeiT processors emit channel-first `pixel_values` matching encoder source expectations: `[batch, channels, height, width]`.
- TrOCR's processor composes image processor plus tokenizer and returns `labels` when text is supplied.
- ViT-GPT2 examples use image feature extractor plus GPT-2 tokenizer; there is no placeholder-token scatter because images are supplied only through encoder hidden states.
- No sampled config requires remote code.

