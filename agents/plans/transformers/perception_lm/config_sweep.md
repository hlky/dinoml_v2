# PerceptionLM Config Sweep Snapshot

Source date: 2026-05-13.

Official checkpoints `facebook/Perception-LM-1B`, `facebook/Perception-LM-3B`,
and `facebook/Perception-LM-8B` are manual-gated on Hugging Face. Raw
`config.json`, processor, and tokenizer files returned 401 without accepted
access. Hub API metadata confirmed public model ids, file names, `model_type:
perception_lm`, `PerceptionLMForConditionalGeneration`, gated status, and BF16
parameter counts. Detailed fields below are from open mirrors and must be
rechecked against official configs after access is granted.

| Source | Basis | Text hidden | Layers | Q heads | KV heads | Head dim | MLP | Vocab | Context | RoPE | Vision arch | Vision dim/depth | Pool |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: |
| `Dhruvil03/Perception-LM-1B-fp16` | open mirror of 1B | 2048 | 16 | 32 | 8 | 64 | 8192 | 128256 | 11520 | llama3 factor 32, theta 500000 | `vit_pe_core_large_patch14_336` | 1024 / 23 | 2 |
| `PIA-SPACE-LAB/Perception-LM-1B` | open mirror of 1B | 2048 | 16 | 32 | 8 | 64 | 8192 | 128256 | 11520 | llama3 factor 32, theta 500000 | `vit_pe_core_large_patch14_336` | 1024 / 23 | 2 |
| `PIA-SPACE-LAB/Perception-LM-3B` | open mirror of 3B | 3072 | 28 | 24 | 8 | 128 | 8192 | 128256 | 11520 | llama3 factor 32, theta 500000 | `vit_pe_core_large_patch14_336` | 1024 / 23 | 2 |
| `Dhruvil03/Perception-LM-8B-Int4-NotBNB` | open 8B mirror/variant | 4096 | 32 | 32 | 8 | 128 | 14336 | 128256 | 11520 | none in mirror config | `vit_pe_core_gigantic_patch14_448` | 1536 / 47 | 2 |
| `Dhruvil03/Perception-LM-1B-Int4bit` | quantized mirror variant | 2048 | 16 | 32 | 8 | 64 | 8192 | 128256 | 11520 | same 1B text config | `vit_pe_core_large_patch14_336` | 1024 / 23 | 2 |

Common open processor snapshot:

```json
{
  "processor_class": "PerceptionLMProcessor",
  "image_processor_type": "PerceptionLMImageProcessorFast",
  "video_processor_type": "PerceptionLMVideoProcessor",
  "patch_size": 14,
  "pooling_ratio": 2,
  "tile_size": 448,
  "max_num_tiles": 36,
  "vision_input_type": "thumb+tile",
  "data_format": "channels_first",
  "do_resize": true,
  "do_center_crop": false,
  "do_rescale": true,
  "rescale_factor": 0.00392156862745098,
  "do_normalize": true,
  "image_mean": [0.5, 0.5, 0.5],
  "image_std": [0.5, 0.5, 0.5],
  "do_convert_rgb": true,
  "video_num_frames": null,
  "video_fps": null,
  "video_do_sample_frames": null
}
```

Official Hub API metadata:

| Official model | Gating | Repo sha seen | Safetensors parameter metadata |
| --- | --- | --- | --- |
| `facebook/Perception-LM-1B` | manual | `2b1a854663b80d6c8b9a10e4b229be97c7f6be1f` | BF16 1,533,524,992 |
| `facebook/Perception-LM-3B` | manual | `31e3665b544e2dbac215e80200923139fb398975` | BF16 3,516,753,920 |
| `facebook/Perception-LM-8B` | manual | `969497c228e880aacb82693813f3169330bec9c8` | BF16 9,794,260,736 |

