# PI0 Hub Config Sweep Snapshot

Fetched on 2026-05-13 via Hugging Face Hub API/raw URLs. These are LeRobot policy configs, not native Transformers `model_type="pi0"` configs.

| Model | Hub sha | Gated | Config basis | Operator-significant facts |
| --- | --- | --- | --- | --- |
| [`lerobot/pi0_base`](https://huggingface.co/lerobot/pi0_base) | `26b99b9439acb1e352439e34ee9c67af0d76efa3` | no | LeRobot `config.json` | 3 visual inputs shaped `[3,224,224]`, state `[32]`, action `[32]`, `paligemma_variant="gemma_2b"`, `action_expert_variant="gemma_300m"`, `dtype="float32"`, chunk/action steps 50, tokenizer max length 48 |
| [`lerobot/pi0_libero_base`](https://huggingface.co/lerobot/pi0_libero_base) | `1dc27a57cf4b54c6fb138ed80a97da150e812e76` | no | LeRobot `config.json` | 2 visual inputs declared `[3,256,256]` but `image_resolution=[224,224]`, state `[8]`, action `[7]`, `empty_cameras=1`, chunk 50, `n_action_steps=10`, dtype float32 |
| [`lerobot/pi0_libero_finetuned_v044`](https://huggingface.co/lerobot/pi0_libero_finetuned_v044) | `45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f` | no | LeRobot `config.json` | 2 visual inputs `[3,256,256]` plus one empty camera `[3,224,224]`, state `[8]`, action `[7]`, dtype bfloat16, gradient checkpointing/compile flags true, chunk/action steps 50 |
| [`lerobot/pi0_old`](https://huggingface.co/lerobot/pi0_old) | `e4ed526af508e58f6008b29e9e48f1098278fdb5` | no | Older LeRobot `config.json` | 3 visual inputs `[3,480,640]` resized/padded to `[224,224]`, state/action `[6]`, `proj_width=1024`, `attention_implementation="eager"`, CPU device in config |
| [`google/paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224) | `35e4f46485b4d07967e7e9935bc3786aad50687c` | manual | Hub API metadata only; raw config requires license/auth | Official PaliGemma backbone checkpoint is gated. Access would resolve exact backbone `config.json`, tokenizer, and preprocessor files. PI0 source defaults match the PaliGemma/Gemma/SigLIP family shape, but exact checkpoint configs should be verified with authenticated access before weight import. |

Hub metadata notes:

- `lerobot/pi0_base` safetensors metadata reports 3,501,372,176 F32 parameters.
- `lerobot/pi0_libero_base` reports 4,335,308 F32 plus 3,497,036,912 BF16 parameters.
- `lerobot/pi0_libero_finetuned_v044` reports 4,335,264 F32 plus 3,497,036,912 BF16 parameters.
- The LeRobot repos include `policy_preprocessor.json` and `policy_postprocessor.json`; `pi0_base` preprocessing includes tokenizer `google/paligemma-3b-pt-224`, `max_length=48`, device transfer, and normalizer/unnormalizer steps.

