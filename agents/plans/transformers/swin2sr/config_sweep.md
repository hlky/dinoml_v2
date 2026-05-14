# Swin2SR Config Sweep

Source: official Hugging Face repositories, fetched from `https://huggingface.co/{model_id}/raw/main/config.json` and `preprocessor_config.json` on 2026-05-13.

| Model id | Access | image_size | embed_dim | depths | heads | window | upsampler | upscale | preprocessor |
| --- | --- | ---: | ---: | --- | --- | ---: | --- | ---: | --- |
| `caidas/swin2SR-classical-sr-x2-64` | 200 | 64 | 180 | `[6,6,6,6,6,6]` | `[6,6,6,6,6,6]` | 8 | `pixelshuffle` | 2 | `do_rescale=true`, `rescale_factor=1/255`, `do_pad=true`, `pad_size=8` |
| `caidas/swin2SR-classical-sr-x4-64` | 200 | 64 | 180 | `[6,6,6,6,6,6]` | `[6,6,6,6,6,6]` | 8 | `pixelshuffle` | 4 | same |
| `caidas/swin2SR-compressed-sr-x4-48` | 200 | 48 | 180 | `[6,6,6,6,6,6]` | `[6,6,6,6,6,6]` | 8 | `pixelshuffle_aux` | 4 | same |
| `caidas/swin2SR-lightweight-x2-64` | 200 | 64 | 60 | `[6,6,6,6]` | `[6,6,6,6]` | 8 | `pixelshuffledirect` | 2 | same |
| `caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr` | 200 | 64 | 180 | `[6,6,6,6,6,6]` | `[6,6,6,6,6,6]` | 8 | `nearest+conv` | 4 | same |

All sampled configs include `model_type="swin2sr"`, `architectures=["Swin2SRForImageSuperResolution"]`, `patch_size=1`, `mlp_ratio=2.0`, `qkv_bias=true`, `resi_connection="1conv"`, `hidden_act="gelu"`, `torch_dtype="float32"`, `img_range=1.0`, `use_absolute_embeddings=false`, and `transformers_version="4.26.0.dev0"`.
