# MGP-STR config snapshots

Source basis: Hugging Face Hub raw files fetched 2026-05-13; Transformers source checkout `X:/H/transformers` at `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

## alibaba-damo/mgp-str-base

Config URL: <https://huggingface.co/alibaba-damo/mgp-str-base/raw/main/config.json>

```json
{
  "_name_or_path": "alibaba-damo/mgp-str-base",
  "architectures": ["MGPSTRModel"],
  "image_size": [32, 128],
  "patch_size": 4,
  "num_channels": 3,
  "max_token_length": 27,
  "num_character_labels": 38,
  "num_bpe_labels": 50257,
  "num_wordpiece_labels": 30522,
  "hidden_size": 768,
  "num_hidden_layers": 12,
  "num_attention_heads": 12,
  "mlp_ratio": 4,
  "qkv_bias": true,
  "drop_rate": 0.0,
  "attn_drop_rate": 0.0,
  "drop_path_rate": 0.0,
  "output_a3_attentions": false,
  "model_type": "mgp-str",
  "torch_dtype": "float32"
}
```

Preprocessor URL: <https://huggingface.co/alibaba-damo/mgp-str-base/raw/main/preprocessor_config.json>

```json
{
  "do_normalize": false,
  "do_resize": true,
  "feature_extractor_type": "ViTFeatureExtractor",
  "resample": 3,
  "size": {"height": 32, "width": 128}
}
```

Character vocab URL: <https://huggingface.co/alibaba-damo/mgp-str-base/raw/main/vocab.json>

```json
{"[GO]": 0, "[s]": 1, "0": 2, "1": 3, "...": "...", "z": 37}
```

## hf-tiny-model-private/tiny-random-MgpstrForSceneTextRecognition

Config URL: <https://huggingface.co/hf-tiny-model-private/tiny-random-MgpstrForSceneTextRecognition/raw/main/config.json>

```json
{
  "architectures": ["MgpstrForSceneTextRecognition"],
  "hidden_size": 32,
  "num_hidden_layers": 5,
  "num_attention_heads": 4,
  "num_bpe_labels": 99,
  "num_wordpiece_labels": 99,
  "image_size": [32, 128],
  "patch_size": 4,
  "max_token_length": 27,
  "mlp_ratio": 4.0,
  "model_type": "mgp-str",
  "torch_dtype": "float32",
  "transformers_version": "4.28.0.dev0"
}
```

## onnx-community/mgp-str-base mirror

Config URL: <https://huggingface.co/onnx-community/mgp-str-base/raw/main/config.json>

Same neural dimensions as `alibaba-damo/mgp-str-base`; adds `_attn_implementation_autoset: true` and `transformers_version: "4.46.1"`, neither read by the inspected native `mgp_str` modeling source.

Preprocessor URL: <https://huggingface.co/onnx-community/mgp-str-base/raw/main/preprocessor_config.json>

```json
{
  "do_normalize": false,
  "do_rescale": true,
  "do_resize": true,
  "image_processor_type": "ViTFeatureExtractor",
  "rescale_factor": 0.00392156862745098,
  "size": {"height": 32, "width": 128}
}
```

## onnx-community / internal tiny mirrors

`onnx-community/tiny-random-MgpstrForSceneTextRecognition` matches the tiny/random config above. `onnx-internal-testing/tiny-random-MgpstrForSceneTextRecognition-ONNX` also matches, adding `_attn_implementation_autoset: true` and `transformers_version: "4.48.2"`.

`ml6team/mgp-str-onnx` did not expose a `config.json` at the standard raw path during this audit.
