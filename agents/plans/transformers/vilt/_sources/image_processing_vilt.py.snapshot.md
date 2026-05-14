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
16: import torch
17: from torchvision.transforms.v2 import functional as tvF
18: 
19: from ...image_processing_backends import TorchvisionBackend
20: from ...image_processing_utils import BatchFeature
21: from ...image_transforms import group_images_by_shape, reorder_images
22: from ...image_utils import (
23:     IMAGENET_STANDARD_MEAN,
24:     IMAGENET_STANDARD_STD,
25:     PILImageResampling,
26:     SizeDict,
27:     get_max_height_width,
28: )
29: from ...processing_utils import ImagesKwargs
30: from ...utils import (
31:     TensorType,
32:     auto_docstring,
33: )
34: 
35: 
36: # Set maximum size based on the typical aspect ratio of the COCO dataset
37: MAX_LONGER_EDGE = 1333
38: MAX_SHORTER_EDGE = 800
39: 
40: 
41: class ViltImageProcessorKwargs(ImagesKwargs, total=False):
42:     r"""
43:     size_divisor (`int`, *optional*, defaults to `self.size_divisor`):
44:         The size by which to make sure both the height and width can be divided. Only has an effect if `do_resize`
45:         is set to `True`.
46:     """
47: 
48:     size_divisor: int
49: 
50: 
51: @auto_docstring
52: class ViltImageProcessor(TorchvisionBackend):
53:     valid_kwargs = ViltImageProcessorKwargs
54:     resample = PILImageResampling.BICUBIC
55:     image_mean = IMAGENET_STANDARD_MEAN
56:     image_std = IMAGENET_STANDARD_STD
57:     size = {"shortest_edge": 384}
58:     do_resize = True
59:     do_rescale = True
60:     do_normalize = True
61:     size_divisor = 32
62:     do_pad = True
63:     default_to_square = False
64:     model_input_names = ["pixel_values", "pixel_mask"]
65: 
66:     def resize(
67:         self,
68:         images: "torch.Tensor",
69:         size: SizeDict,
70:         resample: "PILImageResampling | tvF.InterpolationMode | int | None" = None,
71:         size_divisor: int | None = None,
72:     ) -> "torch.Tensor":
73:         """
74:         Resize an image or batch of images to specified size.
75: 
76:         Args:
77:             images (`torch.Tensor`): Image or batch of images to resize.
78:             size (`SizeDict`): Size dictionary with shortest_edge key.
79:             resample (`PILImageResampling | tvF.InterpolationMode | int`, *optional*): Interpolation method to use.
80:             size_divisor (`int`, *optional*): Value to ensure height/width are divisible by.
81: 
82:         Returns:
83:             `torch.Tensor`: Resized image or batch of images.
84:         """
85: 
86:         # Resize with aspect ratio preservation
87:         shorter = size.shortest_edge
88:         longer = int(MAX_LONGER_EDGE / MAX_SHORTER_EDGE * shorter)
89: 
90:         heights = images.shape[-2]
91:         widths = images.shape[-1]
92: 
93:         # Determine the new dimensions
94:         if heights < widths:
95:             new_heights = shorter
96:             new_widths = widths * (shorter / heights)
97:         else:
98:             new_heights = heights * (shorter / widths)
99:             new_widths = shorter
100: 
101:         # Check if the longer side exceeds max size
102:         if max(new_heights, new_widths) > longer:
103:             scale = longer / max(new_heights, new_widths)
104:             new_heights = new_heights * scale
105:             new_widths = new_widths * scale
106: 
107:         new_heights = int(new_heights + 0.5)
108:         new_widths = int(new_widths + 0.5)
109: 
110:         # Make dimensions divisible by size_divisor
111:         if size_divisor is not None:
112:             new_heights = new_heights // size_divisor * size_divisor
113:             new_widths = new_widths // size_divisor * size_divisor
114: 
115:         # Resize the image
116:         return super().resize(images, SizeDict(height=new_heights, width=new_widths), resample=resample)
117: 
118:     def _pad_batch(
119:         self,
120:         images: list["torch.Tensor"],
121:         return_tensors: str | TensorType | None,
122:         disable_grouping: bool | None,
123:     ) -> tuple:
124:         """
125:         Pad a batch of images to the same size based on the maximum dimensions.
126: 
127:         Args:
128:             images (`list[torch.Tensor]`): List of images to pad.
129:             return_tensors (`str` or `TensorType`, *optional*): The type of tensors to return.
130: 
131:         Returns:
132:             `tuple`: Tuple containing padded images and pixel masks.
133:         """
134:         # Calculate global maximum dimensions across all images
135:         max_size = get_max_height_width(images)
136: 
137:         # Group images by shape before padding
138:         grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
139:         processed_images = {}
140:         processed_masks = {}
141: 
142:         for shape, stacked_images in grouped_images.items():
143:             # Create mask template for efficient masking
144:             if return_tensors == "pt" and len(stacked_images) > 0:
145:                 device = stacked_images.device
146:                 mask_template = torch.zeros(max_size, dtype=torch.int64, device=device)
147: 
148:             original_size = stacked_images.shape[-2:]
149:             needs_padding = original_size[0] != max_size[0] or original_size[1] != max_size[1]
150: 
151:             if needs_padding:
152:                 padding_bottom = max_size[0] - original_size[0]
153:                 padding_right = max_size[1] - original_size[1]
154:                 padding = [0, 0, padding_right, padding_bottom]
155: 
156:                 padded_images = tvF.pad(stacked_images, padding, fill=0)
157:                 pixel_mask = mask_template.clone()
158:                 pixel_mask[: original_size[0], : original_size[1]].fill_(1)
159:                 pixel_masks = pixel_mask.unsqueeze(0).repeat(stacked_images.shape[0], 1, 1)
160:             else:
161:                 padded_images = stacked_images
162:                 pixel_masks = torch.ones(
163:                     (stacked_images.shape[0], max_size[0], max_size[1]),
164:                     dtype=torch.int64,
165:                     device=stacked_images.device,
166:                 )
167: 
168:             # Store processed group
169:             processed_images[shape] = padded_images
170:             processed_masks[shape] = pixel_masks
171: 
172:         # Reorder images back to original order
173:         padded_images = reorder_images(processed_images, grouped_images_index)
174:         pixel_masks = reorder_images(processed_masks, grouped_images_index)
175: 
176:         return padded_images, pixel_masks
177: 
178:     def _preprocess(
179:         self,
180:         images: list["torch.Tensor"],
181:         do_resize: bool,
182:         size: SizeDict,
183:         resample: "PILImageResampling | tvF.InterpolationMode | int | None",
184:         do_rescale: bool,
185:         rescale_factor: float,
186:         do_normalize: bool,
187:         image_mean: float | list[float] | None,
188:         image_std: float | list[float] | None,
189:         do_pad: bool | None,
190:         disable_grouping: bool | None,
191:         return_tensors: str | TensorType | None,
192:         size_divisor: int | None = None,
193:         **kwargs,
194:     ) -> BatchFeature:
195:         # Group images by size for batched resizing
196:         grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
197:         resized_images_grouped = {}
198: 
199:         for shape, stacked_images in grouped_images.items():
200:             if do_resize:
201:                 stacked_images = self.resize(stacked_images, size, resample, size_divisor)
202:             resized_images_grouped[shape] = stacked_images
203:         resized_images = reorder_images(resized_images_grouped, grouped_images_index)
204: 
205:         # Group images by size for further processing
206:         grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
207:         processed_images_grouped = {}
208: 
209:         for shape, stacked_images in grouped_images.items():
210:             # Fused rescale and normalize
211:             stacked_images = self.rescale_and_normalize(
212:                 stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
213:             )
214:             processed_images_grouped[shape] = stacked_images
215: 
216:         processed_images = reorder_images(processed_images_grouped, grouped_images_index)
217: 
218:         # Handle padding if required
219:         data = {}
220:         if do_pad:
221:             pixel_values, pixel_mask = self._pad_batch(
222:                 processed_images, return_tensors, disable_grouping=disable_grouping
223:             )
224:             data = {"pixel_values": pixel_values, "pixel_mask": pixel_mask}
225:         else:
226:             # If no padding, just return the processed images
227:             if return_tensors == "pt":
228:                 processed_images = torch.stack(processed_images)
229:             data = {"pixel_values": processed_images}
230: 
231:         return BatchFeature(data=data, tensor_type=return_tensors)
232: 
233: 
234: __all__ = ["ViltImageProcessor"]