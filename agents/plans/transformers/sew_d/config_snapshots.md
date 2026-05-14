# SEW-D Config Snapshots

Fetched on 2026-05-13 from public Hugging Face model repositories.

## asapp/sew-d-tiny-100k

- URL: https://huggingface.co/asapp/sew-d-tiny-100k/raw/main/config.json
- Architecture: `SEWDModel`
- Task metadata: feature extraction
- Safetensors metadata: total 24,115,103 parameters from HF API
- Key dimensions: `hidden_size=384`, `num_hidden_layers=12`, `num_attention_heads=6`, `intermediate_size=1536`, `conv_dim=[64,128,128,128,128,256,256,256,256,512,512,512,512]`, `num_conv_pos_embeddings=31`, `squeeze_factor=2`
- Preprocessor: `sampling_rate=16000`, `feature_size=1`, `do_normalize=true`, `padding_side=right`, `padding_value=0`, `return_attention_mask=false`

## asapp/sew-d-tiny-100k-ft-ls100h

- URL: https://huggingface.co/asapp/sew-d-tiny-100k-ft-ls100h/raw/main/config.json
- Architecture: `SEWDForCTC`
- Task metadata: automatic speech recognition
- Safetensors metadata: total 24,127,423 parameters from HF API
- Key dimensions: `hidden_size=384`, `num_hidden_layers=12`, `num_attention_heads=6`, `intermediate_size=1536`, `vocab_size=32`, `num_conv_pos_embeddings=31`, `squeeze_factor=2`
- CTC tokenizer vocab: `<pad>=0`, `<s>=1`, `</s>=2`, `<unk>=3`, `|=4`, uppercase letters plus apostrophe through id 31
- Preprocessor: `sampling_rate=16000`, `feature_size=1`, `do_normalize=true`, `padding_side=right`, `padding_value=0`, `return_attention_mask=false`

## asapp/sew-d-small-100k

- URL: https://huggingface.co/asapp/sew-d-small-100k/raw/main/config.json
- Architecture: `SEWDModel`
- Task metadata: feature extraction
- Safetensors: not present in HF API; repository has `pytorch_model.bin`
- Key dimensions: `hidden_size=512`, `num_hidden_layers=12`, `num_attention_heads=8`, `intermediate_size=2048`, `conv_dim=[64,128,128,128,128,256,256,256,256,512,512,512,512]`, `num_conv_pos_embeddings=31`, `squeeze_factor=2`
- Preprocessor: `sampling_rate=16000`, `feature_size=1`, `do_normalize=true`, `padding_side=right`, `padding_value=0`, `return_attention_mask=false`

## asapp/sew-d-mid-400k

- URL: https://huggingface.co/asapp/sew-d-mid-400k/raw/main/config.json
- Architecture: `SEWDModel`
- Task metadata: feature extraction
- Safetensors: not present in HF API; repository has `pytorch_model.bin`
- Key dimensions: `hidden_size=512`, `num_hidden_layers=24`, `num_attention_heads=8`, `intermediate_size=2048`, `conv_dim=[64,128,128,128,128,256,256,256,256,512,512,512,512]`, `num_conv_pos_embeddings=31`, `squeeze_factor=2`
- Preprocessor: `sampling_rate=16000`, `feature_size=1`, `do_normalize=true`, `padding_side=right`, `padding_value=0`, `return_attention_mask=false`

## asapp/sew-d-mid-k127-400k-ft-ls100h

- URL: https://huggingface.co/asapp/sew-d-mid-k127-400k-ft-ls100h/raw/main/config.json
- Architecture: `SEWDForCTC`
- Task metadata: automatic speech recognition
- Safetensors metadata: total 80,389,023 parameters from HF API
- Key dimensions: `hidden_size=512`, `num_hidden_layers=24`, `num_attention_heads=8`, `intermediate_size=2048`, `vocab_size=32`, `num_conv_pos_embeddings=127`, `squeeze_factor=2`
- Preprocessor: `sampling_rate=16000`, `feature_size=1`, `do_normalize=true`, `padding_side=right`, `padding_value=0`, `return_attention_mask=false`

## asapp/sew-d-base-plus-400k-ft-ls100h

- URL: https://huggingface.co/asapp/sew-d-base-plus-400k-ft-ls100h/raw/main/config.json
- Architecture: `SEWDForCTC`
- Task metadata: automatic speech recognition
- Safetensors metadata: total 177,003,711 parameters from HF API
- Key dimensions: `hidden_size=768`, `num_hidden_layers=24`, `num_attention_heads=12`, `intermediate_size=3072`, `vocab_size=32`, `conv_dim=[96,192,192,192,192,384,384,384,384,768,768,768,768]`, `num_conv_pos_embeddings=31`, `squeeze_factor=2`
- Preprocessor: `sampling_rate=16000`, `feature_size=1`, `do_normalize=true`, `padding_side=right`, `padding_value=0`, `return_attention_mask=false`

