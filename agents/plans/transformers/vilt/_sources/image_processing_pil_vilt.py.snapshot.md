1: # Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
14: """Image processor class for Vilt."""
15: 
16: from collections.abc import Iterable
17: from typing import Any
18: 
19: import numpy as np
20: 
21: from ...image_processing_backends import PilBackend
22: from ...image_processing_utils import BatchFeature
23: from ...image_transforms import PaddingMode, pad
24: from ...image_utils import (
25:     IMAGENET_STANDARD_MEAN,
26:     IMAGENET_STANDARD_STD,
27:     ChannelDimension,
28:     PILImageResampling,
29:     SizeDict,
30:     get_image_size,
31:     get_max_height_width,
32: )
33: from ...processing_utils import ImagesKwargs
34: from ...utils import (
35:     TensorType,
36:     auto_docstring,
37: )
38: 
39: 
40: # Set maximum size based on the typical aspect ratio of the COCO dataset
41: MAX_LONGER_EDGE = 1333
42: MAX_SHORTER_EDGE = 800
43: 
44: 
45: def max_across_indices(values: Iterable[Any]) -> list[Any]:
46:     """
47:     Return the maximum value across all indices of an iterable of values.
48:     """
49:     return [max(values_i) for values_i in zip(*values)]
50: 
51: 
52: def make_pixel_mask(
53:     image: np.ndarray, output_size: tuple[int, int], input_data_format: str | ChannelDimension | None = None
54: ) -> np.ndarray:
55:     """
56:     Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.
57: 
58:     Args:
59:         image (`np.ndarray`):
60:             Image to make the pixel mask for.
61:         output_size (`tuple[int, int]`):
62:             Output size of the mask.
63:     """
64:     input_height, input_width = get_image_size(image, channel_dim=input_data_format)
65:     mask = np.zeros(output_size, dtype=np.int64)
66:     mask[:input_height, :input_width] = 1
67:     return mask
68: 
69: 
70: def get_resize_output_image_size(
71:     input_image: np.ndarray,
72:     shorter: int = 800,
73:     longer: int = 1333,
74:     size_divisor: int = 32,
75:     input_data_format: str | ChannelDimension | None = None,
76: ) -> tuple[int, int]:
77:     input_height, input_width = get_image_size(input_image, input_data_format)
78:     min_size, max_size = shorter, longer
79: 
80:     scale = min_size / min(input_height, input_width)
81: 
82:     if input_height < input_width:
83:         new_height = min_size
84:         new_width = scale * input_width
85:     else:
86:         new_height = scale * input_height
87:         new_width = min_size
88: 
89:     if max(new_height, new_width) > max_size:
90:         scale = max_size / max(new_height, new_width)
91:         new_height = scale * new_height
92:         new_width = scale * new_width
93: 
94:     new_height, new_width = int(new_height + 0.5), int(new_width + 0.5)
95:     new_height = new_height // size_divisor * size_divisor
96:     new_width = new_width // size_divisor * size_divisor
97: 
98:     return new_height, new_width
99: 
100: 
101: # Adapted from transformers.models.vilt.image_processing_vilt.ViltImageProcessorKwargs
102: class ViltImageProcessorKwargs(ImagesKwargs, total=False):
103:     r"""
104:     size_divisor (`int`, *optional*, defaults to `self.size_divisor`):
105:         The size by which to make sure both the height and width can be divided. Only has an effect if `do_resize`
106:         is set to `True`.
107:     """
108: 
109:     size_divisor: int
110: 
111: 
112: @auto_docstring
113: class ViltImageProcessorPil(PilBackend):
114:     valid_kwargs = ViltImageProcessorKwargs
115:     resample = PILImageResampling.BICUBIC
116:     image_mean = IMAGENET_STANDARD_MEAN
117:     image_std = IMAGENET_STANDARD_STD
118:     size = {"shortest_edge": 384}
119:     do_resize = True
120:     do_rescale = True
121:     do_normalize = True
122:     size_divisor = 32
123:     do_pad = True
124:     default_to_square = False
125:     model_input_names = ["pixel_values", "pixel_mask"]
126: 
127:     def resize(
128:         self,
129:         image: np.ndarray,
130:         size: SizeDict,
131:         resample: "PILImageResampling | None" = None,
132:         size_divisor: int | None = None,
133:     ) -> np.ndarray:
134:         """
135:         Resize an image to specified size.
136: 
137:         Args:
138:             image (`np.ndarray`): Image to resize.
139:             size (`SizeDict`): Size dictionary with shortest_edge key.
140:             resample (`PILImageResampling | int`, *optional*): Interpolation method to use.
141:             size_divisor (`int`, *optional*): Value to ensure height/width are divisible by.
142: 
143:         Returns:
144:             `np.ndarray`: Resized image.
145:         """
146:         if not hasattr(size, "shortest_edge") or size.shortest_edge is None:
147:             raise ValueError(f"The `size` dictionary must contain the key `shortest_edge`. Got {size}")
148:         shorter = size.shortest_edge
149:         longer = int(MAX_LONGER_EDGE / MAX_SHORTER_EDGE * shorter)
150:         output_size = get_resize_output_image_size(
151:             image,
152:             shorter=shorter,
153:             longer=longer,
154:             size_divisor=size_divisor or self.size_divisor,
155:             input_data_format=ChannelDimension.FIRST,
156:         )
157: 
158:         return super().resize(image, SizeDict(height=output_size[0], width=output_size[1]), resample=resample)
159: 
160:     def _pad_batch(
161:         self,
162:         images: list[np.ndarray],
163:         return_tensors: str | TensorType | None,
164:     ) -> tuple:
165:         """
166:         Pad a batch of images to the same size based on the maximum dimensions.
167: 
168:         Args:
169:             images (`list[np.ndarray]`): List of images to pad.
170:             return_tensors (`str` or `TensorType`, *optional*): The type of tensors to return.
171: 
172:         Returns:
173:             `tuple`: Tuple containing padded images and pixel masks.
174:         """
175:         # Calculate global maximum dimensions across all images
176:         max_size = get_max_height_width(images, input_data_format=ChannelDimension.FIRST)
177: 
178:         padded_images = []
179:         pixel_masks = []
180: 
181:         for image in images:
182:             input_height, input_width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
183:             needs_padding = input_height != max_size[0] or input_width != max_size[1]
184: 
185:             if needs_padding:
186:                 pad_bottom = max_size[0] - input_height
187:                 pad_right = max_size[1] - input_width
188:                 padding = ((0, pad_bottom), (0, pad_right))
189: 
190:                 padded_image = pad(
191:                     image,
192:                     padding,
193:                     mode=PaddingMode.CONSTANT,
194:                     constant_values=0,
195:                     data_format=ChannelDimension.FIRST,
196:                     input_data_format=ChannelDimension.FIRST,
197:                 )
198:                 pixel_mask = make_pixel_mask(image, max_size, input_data_format=ChannelDimension.FIRST)
199:             else:
200:                 padded_image = image
201:                 pixel_mask = np.ones(max_size, dtype=np.int64)
202: 
203:             padded_images.append(padded_image)
204:             pixel_masks.append(pixel_mask)
205: 
206:         return padded_images, pixel_masks
207: 
208:     def _preprocess(
209:         self,
210:         images: list[np.ndarray],
211:         do_resize: bool,
212:         size: SizeDict,
213:         resample: "PILImageResampling | None",
214:         do_rescale: bool,
215:         rescale_factor: float,
216:         do_normalize: bool,
217:         image_mean: float | list[float] | None,
218:         image_std: float | list[float] | None,
219:         do_pad: bool | None,
220:         return_tensors: str | TensorType | None,
221:         size_divisor: int | None = None,
222:         **kwargs,
223:     ) -> BatchFeature:
224:         processed_images = []
225:         for image in images:
226:             if do_resize:
227:                 image = self.resize(image, size, resample, size_divisor)
228:             if do_rescale:
229:                 image = self.rescale(image, rescale_factor)
230:             if do_normalize:
231:                 image = self.normalize(image, image_mean, image_std)
232:             processed_images.append(image)
233: 
234:         # Handle padding if required
235:         data = {}
236:         if do_pad:
237:             pixel_values, pixel_mask = self._pad_batch(processed_images, return_tensors)
238:             data = {"pixel_values": pixel_values, "pixel_mask": pixel_mask}
239:         else:
240:             data = {"pixel_values": processed_images}
241: 
242:         return BatchFeature(data=data, tensor_type=return_tensors)
243: 
244: 
245: __all__ = ["ViltImageProcessorPil"]