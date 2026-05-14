# Representative config sweep

Fetched from Hugging Face raw/API endpoints on 2026-05-13 where accessible. NVIDIA official model repos expose tokenizer files and README conversion instructions but no hosted `config.json`; their effective config is therefore source/conversion-script based unless the user supplies a locally converted checkpoint directory.

| Model id | Config availability | architecture | vocab | H | L | heads | head_dim | FFN | act | max_pos | type_vocab | use_cache | dtype/metadata |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| `nvidia/megatron-bert-cased-345m` | no hosted config | converted local checkpoint | varies; source default 29056 | 1024 | 24 | 16 | 64 | 4096 | gelu | 512 | 2 | true | README says 345M, conversion writes config |
| `nvidia/megatron-bert-uncased-345m` | no hosted config | converted local checkpoint | conversion overrides from checkpoint | 1024 | 24 | 16 | 64 | 4096 | gelu | 512 | 2 | true | tokenizer config `do_lower_case=true` |
| `KBLab/megatron-bert-base-swedish-cased-600k` | hosted `config.json` | `MegatronBertForMaskedLM` | 64128 | 768 | 12 | 12 | 64 | 3072 | gelu | 512 | 2 | true | `torch_dtype=float32`, safetensors F32 |
| `KBLab/megatron-bert-large-swedish-cased-165k` | hosted `config.json` | `MegatronBertForMaskedLM` | 64128 | 1024 | 24 | 16 | 64 | 4096 | gelu | 512 | 2 | true | `torch_dtype=float32`, safetensors F32 |
| `IDEA-CCNL/Erlangshen-MegatronBert-1.3B` | hosted `config.json` | not listed; API `AutoModel` | 21248 | 2048 | 24 | 8 | 256 | 8192 | gelu_new | 512 | 2 | false | Chinese NLU/base encoder repo |
| `IDEA-CCNL/Erlangshen-MegatronBert-3.9B-Chinese` | hosted `config.json` | `MegatronBertForMaskedLM` | 21248 | 2560 | 48 | 40 | 64 | 10240 | gelu | 512 | 2 | false | Chinese fill-mask repo |
| `EMBO/BioMegatron345mUncased` | hosted `config.json` | not listed; API `AutoModel` | 30592 | 1024 | 24 | 16 | 64 | 4096 | gelu_new | 512 | 2 | false | BioMegatron; initializer differs only for init |
| `hf-tiny-model-private/tiny-random-MegatronBertForMaskedLM` | hosted `config.json` | `MegatronBertForMaskedLM` | 1124 | 64 | 5 | 4 | 16 | 37 | gelu | 512 | 16 | true | tiny/debug; has `embedding_size=32` |

Ignored or historical fields for current native source basis:

- `position_embedding_type`: present in several configs but not read by `modeling_megatron_bert.py`; native source always uses absolute learned position embeddings.
- `gradient_checkpointing`: present in some configs but inference should set eval/no-grad; the source has training-only checkpointing behavior.
- `embedding_size`: accepted in tiny/test configs; current Megatron-BERT source only checks for its presence to bypass a divisibility error and does not create a projection from `embedding_size` to `hidden_size`.
- `tokenizer_type`, tokenizer class, language metadata, license, and safetensors metadata are ABI/loading metadata, not graph operator changes.
