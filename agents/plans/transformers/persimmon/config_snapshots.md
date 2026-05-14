# Persimmon Config Snapshots

Source date: 2026-05-13. Hub values fetched from raw Hugging Face files/API.

## adept/persimmon-8b-base

Hub URL: https://huggingface.co/adept/persimmon-8b-base
Hub API: open, not gated, SHA `94dc4e0bb7eeb26ec521eb3f78c36c91f6fe866b`.

```json
{
  "architectures": ["PersimmonForCausalLM"],
  "model_type": "persimmon",
  "hidden_size": 4096,
  "intermediate_size": 16384,
  "num_hidden_layers": 36,
  "num_attention_heads": 64,
  "num_key_value_heads": 64,
  "hidden_act": "relu2",
  "hidden_dropout": 0.6,
  "attention_dropout": 0.0,
  "layer_norm_eps": 1e-5,
  "max_position_embeddings": 16384,
  "qk_layernorm": true,
  "rope_theta": 25000.0,
  "rope_scaling": null,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "vocab_size": 262144,
  "bos_token_id": 71013,
  "eos_token_id": 71013
}
```

Tokenizer config: `tokenizer_class="LlamaTokenizer"`, `add_bos_token=true`, `add_eos_token=false`, BOS token content `|ENDOFTEXT|`, `pad_token=null`.

## adept/persimmon-8b-chat

Hub URL: https://huggingface.co/adept/persimmon-8b-chat
Hub API: open, not gated, SHA `7f1c23bce0eb2a41a5c7417f10ef15405819286e`.

```json
{
  "architectures": ["PersimmonForCausalLM"],
  "model_type": "persimmon",
  "hidden_size": 4096,
  "intermediate_size": 16384,
  "num_hidden_layers": 36,
  "num_attention_heads": 64,
  "num_key_value_heads": 64,
  "hidden_act": "relu2",
  "hidden_dropout": 0.6,
  "attention_dropout": 0.0,
  "layer_norm_eps": 1e-5,
  "max_position_embeddings": 16384,
  "qk_layernorm": true,
  "rope_theta": 25000.0,
  "rope_scaling": null,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "vocab_size": 262144,
  "bos_token_id": 71013,
  "eos_token_id": 71013
}
```

Tokenizer config matches the base checkpoint.

## optimum-intel-internal-testing/tiny-random-PersimmonForCausalLM

Hub URL: https://huggingface.co/optimum-intel-internal-testing/tiny-random-PersimmonForCausalLM

```json
{
  "architectures": ["PersimmonForCausalLM"],
  "model_type": "persimmon",
  "hidden_size": 32,
  "intermediate_size": 37,
  "num_hidden_layers": 2,
  "num_attention_heads": 4,
  "hidden_act": "gelu",
  "hidden_dropout": 0.0,
  "attention_dropout": 0.0,
  "layer_norm_eps": 1e-5,
  "max_position_embeddings": 512,
  "partial_rotary_factor": 0.5,
  "qk_layernorm": true,
  "rope_theta": 25000.0,
  "rope_scaling": null,
  "tie_word_embeddings": false,
  "torch_dtype": "float32",
  "vocab_size": 262144,
  "bos_token_id": 71013,
  "eos_token_id": 2,
  "pad_token_id": 0
}
```

## Out-of-scope derivative observed

`OpenVINO/persimmon-8b-chat-fp16-ov` is tagged as derived from Adept Persimmon but its raw `config.json` has `model_type="gpt_neox"` and `architectures=["GPTNeoXForCausalLM"]`. Route this to a GPT-NeoX/OpenVINO audit, not this Persimmon source-basis report.
