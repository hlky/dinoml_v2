1: # Copyright 2022 The HuggingFace Inc. team. All rights reserved.
2: #
3: # Licensed under the Apache License, Version 2.0 (the "License");
4: # you may not use this file except in compliance with the License.
5: # You may obtain a copy of the License at
6: #
7: #     http://www.apache.org/licenses/LICENSE-2.0
8: #
9: # Unless required by applicable law or agreed to in writing, software
10: # distributed under the License is distributed on an "AS IS" BASIS,
11: # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
12: # See the License for the specific language governing permissions and
13: # limitations under the License.
14: """VilT model configuration"""
15: 
16: from huggingface_hub.dataclasses import strict
17: 
18: from ...configuration_utils import PreTrainedConfig
19: from ...utils import auto_docstring
20: 
21: 
22: @auto_docstring(checkpoint="dandelin/vilt-b32-mlm")
23: @strict
24: class ViltConfig(PreTrainedConfig):
25:     r"""
26:     modality_type_vocab_size (`int`, *optional*, defaults to 2):
27:         The vocabulary size of the modalities passed when calling [`ViltModel`]. This is used after concatenating the
28:         embeddings of the text and image modalities.
29:     max_image_length (`int`, *optional*, defaults to -1):
30:         The maximum number of patches to take as input for the Transformer encoder. If set to a positive integer,
31:         the encoder will sample `max_image_length` patches at maximum. If set to -1, will not be taken into
32:         account.
33:     num_images (`int`, *optional*, defaults to -1):
34:         The number of images to use for natural language visual reasoning. If set to a positive integer, will be
35:         used by [`ViltForImagesAndTextClassification`] for defining the classifier head.
36: 
37:     Example:
38: 
39:     ```python
40:     >>> from transformers import ViLTModel, ViLTConfig
41: 
42:     >>> # Initializing a ViLT dandelin/vilt-b32-mlm style configuration
43:     >>> configuration = ViLTConfig()
44: 
45:     >>> # Initializing a model from the dandelin/vilt-b32-mlm style configuration
46:     >>> model = ViLTModel(configuration)
47: 
48:     >>> # Accessing the model configuration
49:     >>> configuration = model.config
50:     ```"""
51: 
52:     model_type = "vilt"
53: 
54:     vocab_size: int = 30522
55:     type_vocab_size: int = 2
56:     modality_type_vocab_size: int = 2
57:     max_position_embeddings: int = 40
58:     hidden_size: int = 768
59:     num_hidden_layers: int = 12
60:     num_attention_heads: int = 12
61:     intermediate_size: int = 3072
62:     hidden_act: str = "gelu"
63:     hidden_dropout_prob: float | int = 0.0
64:     attention_probs_dropout_prob: float | int = 0.0
65:     initializer_range: float = 0.02
66:     layer_norm_eps: float = 1e-12
67:     image_size: int | list[int] | tuple[int, int] = 384
68:     patch_size: int | list[int] | tuple[int, int] = 32
69:     num_channels: int = 3
70:     qkv_bias: bool = True
71:     max_image_length: int = -1
72:     tie_word_embeddings: bool = True
73:     num_images: int = -1
74:     pad_token_id: int | None = None
75: 
76:     def __post_init__(self, **kwargs):
77:         kwargs.pop("tie_word_embeddings", None)
78:         self.tie_word_embeddings = True  # force it
79:         super().__post_init__(**kwargs)
80: 
81: 
82: __all__ = ["ViltConfig"]