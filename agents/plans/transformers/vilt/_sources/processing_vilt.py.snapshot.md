1: # Copyright 2022 The HuggingFace Inc. team.
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
14: """
15: Processor class for ViLT.
16: """
17: 
18: from ...processing_utils import ProcessingKwargs, ProcessorMixin
19: from ...utils import auto_docstring
20: 
21: 
22: class ViltProcessorKwargs(ProcessingKwargs, total=False):
23:     _defaults = {
24:         "text_kwargs": {
25:             "add_special_tokens": True,
26:             "padding": False,
27:             "stride": 0,
28:             "return_overflowing_tokens": False,
29:             "return_special_tokens_mask": False,
30:             "return_offsets_mapping": False,
31:             "return_length": False,
32:             "verbose": True,
33:         },
34:     }
35: 
36: 
37: @auto_docstring
38: class ViltProcessor(ProcessorMixin):
39:     valid_processor_kwargs = ViltProcessorKwargs
40: 
41:     def __init__(self, image_processor=None, tokenizer=None, **kwargs):
42:         super().__init__(image_processor, tokenizer)
43: 
44: 
45: __all__ = ["ViltProcessor"]