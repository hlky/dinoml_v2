# Representative config snapshots

Fetched from Hugging Face raw URLs on 2026-05-13. All fetched configs were public; no gated/401 gaps were observed.

## microsoft/udop-large

Source: <https://huggingface.co/microsoft/udop-large/raw/main/config.json>

```json
{
  "architectures": ["UdopForConditionalGeneration"],
  "d_ff": 4096,
  "d_kv": 64,
  "d_model": 1024,
  "decoder_start_token_id": 0,
  "feed_forward_proj": "relu",
  "image_size": 224,
  "max_2d_position_embeddings": 1024,
  "num_channels": 3,
  "num_decoder_layers": 24,
  "num_heads": 16,
  "num_layers": 24,
  "patch_size": 16,
  "relative_attention_max_distance": 128,
  "relative_attention_num_buckets": 32,
  "relative_bias_args": [{"type": "1d"}, {"type": "horizontal"}, {"type": "vertical"}],
  "torch_dtype": "float32",
  "use_cache": true,
  "vocab_size": 33201
}
```

Processor source: <https://huggingface.co/microsoft/udop-large/raw/main/preprocessor_config.json>

```json
{
  "image_processor_type": "LayoutLMv3ImageProcessor",
  "processor_class": "UdopProcessor",
  "apply_ocr": true,
  "do_resize": true,
  "do_rescale": true,
  "do_normalize": true,
  "rescale_factor": 0.00392156862745098,
  "image_mean": [0.485, 0.456, 0.406],
  "image_std": [0.229, 0.224, 0.225],
  "size": {"height": 224, "width": 224},
  "tesseract_config": ""
}
```

## microsoft/udop-large-512

Source: <https://huggingface.co/microsoft/udop-large-512/raw/main/config.json>

Same neural dimensions as `microsoft/udop-large`, except `image_size=512`; with `patch_size=16`, this changes patch count from `14*14=196` to `32*32=1024`.

Processor source: <https://huggingface.co/microsoft/udop-large-512/raw/main/preprocessor_config.json>

Processor size is `{"height": 512, "width": 512}` with the same LayoutLMv3 image processor, OCR, rescale, and ImageNet normalization settings.

## microsoft/udop-large-512-300k

Source: <https://huggingface.co/microsoft/udop-large-512-300k/raw/main/config.json>

Same operator-significant config as `microsoft/udop-large-512`: 24 encoder and 24 decoder layers, `d_model=1024`, `d_ff=4096`, `num_heads=16`, `d_kv=64`, `image_size=512`, `patch_size=16`, `feed_forward_proj="relu"`.

## nielsr/udop-test and nielsr/udop-large

Sources:

- <https://huggingface.co/nielsr/udop-test/raw/main/config.json>
- <https://huggingface.co/nielsr/udop-large/raw/main/config.json>

Both mirror the 224-size large topology. Their older preprocessor configs use historical `"image_processor_type": "UdopImageProcessor"` even though the pinned in-library `UdopProcessor` now documents and composes `LayoutLMv3ImageProcessor`; route these as historical metadata, not a separate neural implementation.
