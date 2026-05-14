# NLLB-MoE Config Sweep Snapshot

Fetched 2026-05-13 by direct Hugging Face raw/API URLs. These are metadata
snapshots for the audit report, not DinoML runtime inputs.

| Model id | Access | Source | Key operator-significant fields |
| --- | --- | --- | --- |
| `facebook/nllb-moe-54b` | 200 | `https://huggingface.co/facebook/nllb-moe-54b/raw/main/config.json` | `model_type="nllb-moe"`, `architectures=["NllbMoeModel"]`, `d_model=2048`, encoder/decoder layers `24/24`, heads `16/16`, FFN `8192/8192`, sparse step `4/4`, `num_experts=128`, `expert_capacity=64`, vocab `256206`, max positions `1024`, `router_dtype="float32"`, `second_expert_policy="all"`, `torch_dtype="float32"`. |
| `facebook/nllb-moe-54b` generation | 200 | `https://huggingface.co/facebook/nllb-moe-54b/raw/main/generation_config.json` | `decoder_start_token_id=2`, `bos=0`, `eos=2`, `pad=1`, `max_new_tokens=200`, `num_beams=4`. Target language still must be supplied as `forced_bos_token_id` by tokenizer/generation caller. |
| `hf-internal-testing/random-nllb-moe-2-experts` | 200 | `https://huggingface.co/hf-internal-testing/random-nllb-moe-2-experts/raw/main/config.json` | Same 54B dimensions in config, but `num_experts=2`; used by Transformers integration tests as a random/debug checkpoint. |
| `hf-tiny-model-private/tiny-random-NllbMoeForConditionalGeneration` | 200 | `https://huggingface.co/hf-tiny-model-private/tiny-random-NllbMoeForConditionalGeneration/raw/main/config.json` | Tiny public test config: `model_type="nllb_moe"` underscore spelling, `architectures=["NllbMoeForConditionalGeneration"]`, `d_model=16`, layers `4/4`, heads `4/4`, FFN `4/4`, sparse step encoder `2`, decoder `1`, `num_experts=4`, `expert_capacity=100`, vocab `256204`, max positions `20`. |
| `hf-tiny-model-private/tiny-random-NllbMoeModel` | 200 | `https://huggingface.co/hf-tiny-model-private/tiny-random-NllbMoeModel/raw/main/config.json` | Same tiny dimensions as conditional-generation tiny checkpoint, but architecture is base `NllbMoeModel`. |
| `madatnlp/nllb-moe-54b-8bit` | 200 | `https://huggingface.co/madatnlp/nllb-moe-54b-8bit/raw/main/config.json` | Community mirror/derivative of 54B with `architectures=["NllbMoeForConditionalGeneration"]`, `torch_dtype="float16"`, bitsandbytes-style `quantization_config.load_in_8bit=true`. Not an in-library native packed-kernel path. |
| `KnutJaegersberg/nllb-moe-54b-4bit` | 200 | `https://huggingface.co/KnutJaegersberg/nllb-moe-54b-4bit/raw/main/config.json` | Community mirror/derivative of 54B with `quantization_config.load_in_4bit=true`, `quant_method="bitsandbytes"`, `torch_dtype="float16"`. Not an in-library native packed-kernel path. |
| `ArthurZ/nllb-moe-128` | 401 | `https://huggingface.co/ArthurZ/nllb-moe-128/raw/main/config.json` and API | Unauthorized; audit cannot inspect dimensions or weight metadata without access. |

