### embeddings_visual_stitch (modeling_vilt.py:67-222)
67: class ViltEmbeddings(nn.Module):
68:     """
69:     Construct the text and patch embeddings.
70: 
71:     Text embeddings are equivalent to BERT embeddings.
72: 
73:     Patch embeddings are equivalent to ViT embeddings.
74:     """
75: 
76:     def __init__(self, config):
77:         super().__init__()
78: 
79:         # text embeddings
80:         self.text_embeddings = TextEmbeddings(config)
81:         # patch embeddings
82:         self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
83:         self.patch_embeddings = ViltPatchEmbeddings(config)
84:         num_patches = self.patch_embeddings.num_patches
85:         self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
86:         # modality type (text/patch) embeddings
87:         self.token_type_embeddings = nn.Embedding(config.modality_type_vocab_size, config.hidden_size)
88:         self.dropout = nn.Dropout(config.hidden_dropout_prob)
89:         self.config = config
90: 
91:     def visual_embed(self, pixel_values, pixel_mask, max_image_length=200):
92:         _, _, ph, pw = self.patch_embeddings.projection.weight.shape
93: 
94:         x = self.patch_embeddings(pixel_values)
95:         x_mask = pixel_mask[:, None, :, :].float()
96:         x_mask = nn.functional.interpolate(x_mask, size=(x.shape[2], x.shape[3])).long()
97:         x_h = x_mask[:, 0].sum(dim=1)[:, 0]
98:         x_w = x_mask[:, 0].sum(dim=2)[:, 0]
99: 
100:         batch_size, num_channels, height, width = x.shape
101:         patch_dim = self.config.image_size // self.config.patch_size
102:         spatial_pos = self.position_embeddings[:, 1:, :].transpose(1, 2).view(1, num_channels, patch_dim, patch_dim)
103:         pos_embed = torch.cat(
104:             [
105:                 nn.functional.pad(
106:                     nn.functional.interpolate(
107:                         spatial_pos,
108:                         size=(h, w),
109:                         mode="bilinear",
110:                         align_corners=True,
111:                     ),
112:                     (0, width - w, 0, height - h),
113:                 )
114:                 for h, w in zip(x_h, x_w)
115:             ],
116:             dim=0,
117:         )
118: 
119:         pos_embed = pos_embed.flatten(2).transpose(1, 2)
120:         x = x.flatten(2).transpose(1, 2)
121:         # Set `device` here, otherwise `patch_index` will always be on `CPU` and will fail near the end for torch>=1.13
122:         patch_index = torch.stack(
123:             torch.meshgrid(torch.arange(x_mask.shape[-2]), torch.arange(x_mask.shape[-1]), indexing="ij"), dim=-1
124:         ).to(device=x_mask.device)
125:         patch_index = patch_index[None, None, :, :, :]
126:         patch_index = patch_index.expand(x_mask.shape[0], x_mask.shape[1], -1, -1, -1)
127:         patch_index = patch_index.flatten(1, 3)
128:         x_mask = x_mask.flatten(1)
129: 
130:         if max_image_length < 0 or max_image_length is None or not isinstance(max_image_length, int):
131:             # suppose aug is 800 x 1333, then, maximum effective res is 800 x 1333 (if one side gets bigger, the other will be constrained and be shrunk)
132:             # (800 // self.patch_size) * (1333 // self.patch_size) is the maximum number of patches that single image can get.
133:             # if self.patch_size = 32, 25 * 41 = 1025
134:             # if res is 384 x 640, 12 * 20 = 240
135:             effective_resolution = x_h * x_w
136:             max_image_length = effective_resolution.max()
137:         else:
138:             effective_resolution = x_h * x_w
139:             max_image_length = min(effective_resolution.max(), max_image_length)
140: 
141:         valid_idx = x_mask.nonzero(as_tuple=False)
142:         non_valid_idx = (1 - x_mask).nonzero(as_tuple=False)
143:         unique_rows = valid_idx[:, 0].unique()
144:         valid_row_idx = [valid_idx[valid_idx[:, 0] == u] for u in unique_rows]
145:         non_valid_row_idx = [non_valid_idx[non_valid_idx[:, 0] == u] for u in unique_rows]
146: 
147:         valid_nums = [v.size(0) for v in valid_row_idx]
148:         non_valid_nums = [v.size(0) for v in non_valid_row_idx]
149:         pad_nums = [max_image_length - v for v in valid_nums]
150: 
151:         select = []
152:         for i, (v, nv, p) in enumerate(zip(valid_nums, non_valid_nums, pad_nums)):
153:             if p <= 0:
154:                 valid_choice = torch.multinomial(torch.ones(v).float(), max_image_length)
155:                 select.append(valid_row_idx[i][valid_choice])
156:             else:
157:                 pad_choice = torch.multinomial(torch.ones(nv).float(), p, replacement=True)
158:                 select.append(torch.cat([valid_row_idx[i], non_valid_row_idx[i][pad_choice]], dim=0))
159: 
160:         select = torch.cat(select, dim=0)
161:         x = x[select[:, 0], select[:, 1]].view(batch_size, -1, num_channels)
162:         x_mask = x_mask[select[:, 0], select[:, 1]].view(batch_size, -1)
163:         # `patch_index` should be on the same device as `select`, which is ensured at definition time.
164:         patch_index = patch_index[select[:, 0], select[:, 1]].view(batch_size, -1, 2)
165:         pos_embed = pos_embed[select[:, 0], select[:, 1]].view(batch_size, -1, num_channels)
166: 
167:         cls_tokens = self.cls_token.expand(batch_size, -1, -1)
168:         x = torch.cat((cls_tokens, x), dim=1)
169:         pos_embed = torch.cat(
170:             (self.position_embeddings[:, 0, :][:, None, :].expand(batch_size, -1, -1), pos_embed), dim=1
171:         )
172:         x = x + pos_embed
173:         x = self.dropout(x)
174: 
175:         x_mask = torch.cat([torch.ones(x_mask.shape[0], 1).to(x_mask), x_mask], dim=1)
176: 
177:         return x, x_mask, (patch_index, (height, width))
178: 
179:     def forward(
180:         self,
181:         input_ids,
182:         attention_mask,
183:         token_type_ids,
184:         pixel_values,
185:         pixel_mask,
186:         inputs_embeds,
187:         image_embeds,
188:         image_token_type_idx=1,
189:     ):
190:         # PART 1: text embeddings
191:         text_embeds = self.text_embeddings(
192:             input_ids=input_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
193:         )
194: 
195:         # PART 2: patch embeddings (with interpolated position encodings)
196:         if image_embeds is None:
197:             image_embeds, image_masks, patch_index = self.visual_embed(
198:                 pixel_values, pixel_mask, max_image_length=self.config.max_image_length
199:             )
200:         else:
201:             image_masks = pixel_mask.flatten(1)
202: 
203:         # PART 3: add modality type embeddings
204:         # 0 indicates text, 1 indicates image, 2 is optionally used when a second image is provided (NLVR2)
205:         if image_token_type_idx is None:
206:             image_token_type_idx = 1
207:         text_embeds = text_embeds + self.token_type_embeddings(
208:             torch.zeros_like(attention_mask, dtype=torch.long, device=text_embeds.device)
209:         )
210:         image_embeds = image_embeds + self.token_type_embeddings(
211:             torch.full_like(image_masks, image_token_type_idx, dtype=torch.long, device=text_embeds.device)
212:         )
213: 
214:         # PART 4: concatenate
215:         embeddings = torch.cat([text_embeds, image_embeds], dim=1)
216:         masks = torch.cat([attention_mask, image_masks], dim=1)
217: 
218:         return embeddings, masks
219: 
220: 
221: class TextEmbeddings(nn.Module):
222:     """Construct the embeddings from word, position and token_type embeddings."""

### patch_embeddings (modeling_vilt.py:275-304)
275: class ViltPatchEmbeddings(nn.Module):
276:     """
277:     Image to Patch Embedding.
278:     """
279: 
280:     def __init__(self, config):
281:         super().__init__()
282:         image_size, patch_size = config.image_size, config.patch_size
283:         num_channels, hidden_size = config.num_channels, config.hidden_size
284: 
285:         image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
286:         patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
287:         num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
288:         self.image_size = image_size
289:         self.patch_size = patch_size
290:         self.num_channels = num_channels
291:         self.num_patches = num_patches
292: 
293:         self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
294: 
295:     def forward(self, pixel_values):
296:         batch_size, num_channels, height, width = pixel_values.shape
297:         if num_channels != self.num_channels:
298:             raise ValueError(
299:                 "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
300:             )
301:         target_dtype = self.projection.weight.dtype
302:         x = self.projection(pixel_values.to(dtype=target_dtype))
303:         return x
304: 

### self_attention (modeling_vilt.py:306-356)
306: class ViltSelfAttention(nn.Module):
307:     def __init__(self, config):
308:         super().__init__()
309:         if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
310:             raise ValueError(
311:                 f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
312:                 f"heads {config.num_attention_heads}."
313:             )
314: 
315:         self.num_attention_heads = config.num_attention_heads
316:         self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
317:         self.all_head_size = self.num_attention_heads * self.attention_head_size
318: 
319:         self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
320:         self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
321:         self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
322: 
323:         self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
324: 
325:     def forward(self, hidden_states, attention_mask=None, output_attentions=False):
326:         input_shape = hidden_states.shape[:-1]
327:         hidden_shape = (*input_shape, -1, self.attention_head_size)
328:         query_layer = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
329:         key_layer = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
330:         value_layer = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
331: 
332:         # Take the dot product between "query" and "key" to get the raw attention scores.
333:         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
334:         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
335:         if attention_mask is not None:
336:             # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
337:             attention_scores = attention_scores + attention_mask
338: 
339:         # Normalize the attention scores to probabilities.
340:         attention_probs = nn.Softmax(dim=-1)(attention_scores)
341: 
342:         # This is actually dropping out entire tokens to attend to, which might
343:         # seem a bit unusual, but is taken from the original Transformer paper.
344:         attention_probs = self.dropout(attention_probs)
345: 
346:         context_layer = torch.matmul(attention_probs, value_layer)
347: 
348:         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
349:         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
350:         context_layer = context_layer.view(*new_context_layer_shape)
351: 
352:         outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
353: 
354:         return outputs
355: 
356: 

### encoder_layer (modeling_vilt.py:420-455)
420: class ViltLayer(GradientCheckpointingLayer):
421:     """This corresponds to the Block class in the timm implementation."""
422: 
423:     def __init__(self, config):
424:         super().__init__()
425:         self.chunk_size_feed_forward = config.chunk_size_feed_forward
426:         self.seq_len_dim = 1
427:         self.attention = ViltAttention(config)
428:         self.intermediate = ViltIntermediate(config)
429:         self.output = ViltOutput(config)
430:         self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
431:         self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
432: 
433:     def forward(self, hidden_states, attention_mask=None, output_attentions=False):
434:         self_attention_outputs = self.attention(
435:             self.layernorm_before(hidden_states),  # in ViLT, layernorm is applied before self-attention
436:             attention_mask,
437:             output_attentions=output_attentions,
438:         )
439:         attention_output = self_attention_outputs[0]
440:         outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
441: 
442:         # first residual connection
443:         hidden_states = attention_output + hidden_states.to(attention_output.device)
444: 
445:         # in ViLT, layernorm is also applied after self-attention
446:         layer_output = self.layernorm_after(hidden_states)
447:         layer_output = self.intermediate(layer_output)
448: 
449:         # second residual connection is done here
450:         layer_output = self.output(layer_output, hidden_states)
451: 
452:         outputs = (layer_output,) + outputs
453: 
454:         return outputs
455: 

### model_forward_pooler (modeling_vilt.py:514-649)
514: class ViltModel(ViltPreTrainedModel):
515:     def __init__(self, config, add_pooling_layer=True):
516:         r"""
517:         add_pooling_layer (bool, *optional*, defaults to `True`):
518:             Whether to add a pooling layer
519:         """
520:         super().__init__(config)
521:         self.config = config
522: 
523:         self.embeddings = ViltEmbeddings(config)
524:         self.encoder = ViltEncoder(config)
525: 
526:         self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
527:         self.pooler = ViltPooler(config) if add_pooling_layer else None
528: 
529:         # Initialize weights and apply final processing
530:         self.post_init()
531: 
532:     def get_input_embeddings(self):
533:         return self.embeddings.text_embeddings.word_embeddings
534: 
535:     def set_input_embeddings(self, value):
536:         self.embeddings.text_embeddings.word_embeddings = value
537: 
538:     @auto_docstring
539:     def forward(
540:         self,
541:         input_ids: torch.LongTensor | None = None,
542:         attention_mask: torch.FloatTensor | None = None,
543:         token_type_ids: torch.LongTensor | None = None,
544:         pixel_values: torch.FloatTensor | None = None,
545:         pixel_mask: torch.LongTensor | None = None,
546:         inputs_embeds: torch.FloatTensor | None = None,
547:         image_embeds: torch.FloatTensor | None = None,
548:         image_token_type_idx: int | None = None,
549:         output_attentions: bool | None = None,
550:         output_hidden_states: bool | None = None,
551:         return_dict: bool | None = None,
552:         **kwargs,
553:     ) -> BaseModelOutputWithPooling | tuple[torch.FloatTensor]:
554:         r"""
555:         image_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`, *optional*):
556:             Optionally, instead of passing `pixel_values`, you can choose to directly pass an embedded representation.
557:             This is useful if you want more control over how to convert `pixel_values` into patch embeddings.
558:         image_token_type_idx (`int`, *optional*):
559:             - The token type ids for images.
560: 
561:         Examples:
562: 
563:         ```python
564:         >>> from transformers import ViltProcessor, ViltModel
565:         >>> from PIL import Image
566:         >>> import httpx
567:         >>> from io import BytesIO
568: 
569:         >>> # prepare image and text
570:         >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
571:         >>> with httpx.stream("GET", url) as response:
572:         ...     image = Image.open(BytesIO(response.read()))
573:         >>> text = "hello world"
574: 
575:         >>> processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
576:         >>> model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
577: 
578:         >>> inputs = processor(image, text, return_tensors="pt")
579:         >>> outputs = model(**inputs)
580:         >>> last_hidden_states = outputs.last_hidden_state
581:         ```"""
582:         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
583:         output_hidden_states = (
584:             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
585:         )
586:         return_dict = return_dict if return_dict is not None else self.config.return_dict
587: 
588:         if input_ids is not None and inputs_embeds is not None:
589:             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
590:         elif input_ids is not None:
591:             self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
592:             input_shape = input_ids.size()
593:         elif inputs_embeds is not None:
594:             input_shape = inputs_embeds.size()[:-1]
595:         else:
596:             raise ValueError("You have to specify either input_ids or inputs_embeds")
597: 
598:         text_batch_size, seq_length = input_shape
599:         device = input_ids.device if input_ids is not None else inputs_embeds.device
600: 
601:         if attention_mask is None:
602:             attention_mask = torch.ones(((text_batch_size, seq_length)), device=device)
603: 
604:         if pixel_values is not None and image_embeds is not None:
605:             raise ValueError("You cannot specify both pixel_values and image_embeds at the same time")
606:         elif pixel_values is None and image_embeds is None:
607:             raise ValueError("You have to specify either pixel_values or image_embeds")
608: 
609:         image_batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeds.shape[0]
610:         if image_batch_size != text_batch_size:
611:             raise ValueError("The text inputs and image inputs need to have the same batch size")
612:         if pixel_mask is None:
613:             pixel_mask = torch.ones((image_batch_size, self.config.image_size, self.config.image_size), device=device)
614: 
615:         embedding_output, attention_mask = self.embeddings(
616:             input_ids,
617:             attention_mask,
618:             token_type_ids,
619:             pixel_values,
620:             pixel_mask,
621:             inputs_embeds,
622:             image_embeds,
623:             image_token_type_idx=image_token_type_idx,
624:         )
625: 
626:         # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
627:         # ourselves in which case we just need to make it broadcastable to all heads.
628:         extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
629: 
630:         encoder_outputs = self.encoder(
631:             embedding_output,
632:             attention_mask=extended_attention_mask,
633:             output_attentions=output_attentions,
634:             output_hidden_states=output_hidden_states,
635:             return_dict=return_dict,
636:         )
637:         sequence_output = encoder_outputs[0]
638:         sequence_output = self.layernorm(sequence_output)
639:         pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
640: 
641:         if not return_dict:
642:             return (sequence_output, pooled_output) + encoder_outputs[1:]
643: 
644:         return BaseModelOutputWithPooling(
645:             last_hidden_state=sequence_output,
646:             pooler_output=pooled_output,
647:             hidden_states=encoder_outputs.hidden_states,
648:             attentions=encoder_outputs.attentions,
649:         )

### pooler (modeling_vilt.py:652-667)
652: class ViltPooler(nn.Module):
653:     def __init__(self, config):
654:         super().__init__()
655:         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
656:         self.activation = nn.Tanh()
657: 
658:     def forward(self, hidden_states):
659:         # We "pool" the model by simply taking the hidden state corresponding
660:         # to the first token.
661:         first_token_tensor = hidden_states[:, 0]
662:         pooled_output = self.dense(first_token_tensor)
663:         pooled_output = self.activation(pooled_output)
664:         return pooled_output
665: 
666: 
667: @auto_docstring(

### mlm_head (modeling_vilt.py:672-840)
672: class ViltForMaskedLM(ViltPreTrainedModel):
673:     _tied_weights_keys = {
674:         "mlm_score.decoder.weight": "vilt.embeddings.text_embeddings.word_embeddings.weight",
675:     }
676: 
677:     def __init__(self, config):
678:         super().__init__(config)
679: 
680:         self.vilt = ViltModel(config)
681:         self.mlm_score = ViltMLMHead(config)
682: 
683:         # Initialize weights and apply final processing
684:         self.post_init()
685: 
686:     def get_output_embeddings(self):
687:         return self.mlm_score.decoder
688: 
689:     def set_output_embeddings(self, new_embeddings):
690:         self.mlm_score.decoder = new_embeddings
691:         self.mlm_score.bias = new_embeddings.bias
692: 
693:     @auto_docstring
694:     def forward(
695:         self,
696:         input_ids: torch.LongTensor | None = None,
697:         attention_mask: torch.FloatTensor | None = None,
698:         token_type_ids: torch.LongTensor | None = None,
699:         pixel_values: torch.FloatTensor | None = None,
700:         pixel_mask: torch.LongTensor | None = None,
701:         inputs_embeds: torch.FloatTensor | None = None,
702:         image_embeds: torch.FloatTensor | None = None,
703:         labels: torch.LongTensor | None = None,
704:         output_attentions: bool | None = None,
705:         output_hidden_states: bool | None = None,
706:         return_dict: bool | None = None,
707:         **kwargs,
708:     ) -> MaskedLMOutput | tuple[torch.FloatTensor]:
709:         r"""
710:         image_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`, *optional*):
711:             Optionally, instead of passing `pixel_values`, you can choose to directly pass an embedded representation.
712:             This is useful if you want more control over how to convert `pixel_values` into patch embeddings.
713:         labels (*torch.LongTensor* of shape *(batch_size, sequence_length)*, *optional*):
714:             Labels for computing the masked language modeling loss. Indices should be in *[-100, 0, ...,
715:             config.vocab_size]* (see *input_ids* docstring) Tokens with indices set to *-100* are ignored (masked), the
716:             loss is only computed for the tokens with labels in *[0, ..., config.vocab_size]*
717: 
718:         Examples:
719: 
720:         ```python
721:         >>> from transformers import ViltProcessor, ViltForMaskedLM
722:         >>> import httpx
723:         >>> from io import BytesIO
724:         >>> from PIL import Image
725:         >>> import re
726:         >>> import torch
727: 
728:         >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
729:         >>> with httpx.stream("GET", url) as response:
730:         ...     image = Image.open(BytesIO(response.read()))
731:         >>> text = "a bunch of [MASK] laying on a [MASK]."
732: 
733:         >>> processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
734:         >>> model = ViltForMaskedLM.from_pretrained("dandelin/vilt-b32-mlm")
735: 
736:         >>> # prepare inputs
737:         >>> encoding = processor(image, text, return_tensors="pt")
738: 
739:         >>> # forward pass
740:         >>> outputs = model(**encoding)
741: 
742:         >>> tl = len(re.findall("\[MASK\]", text))
743:         >>> inferred_token = [text]
744: 
745:         >>> # gradually fill in the MASK tokens, one by one
746:         >>> with torch.no_grad():
747:         ...     for i in range(tl):
748:         ...         encoded = processor.tokenizer(inferred_token)
749:         ...         input_ids = torch.tensor(encoded.input_ids)
750:         ...         encoded = encoded["input_ids"][0][1:-1]
751:         ...         outputs = model(input_ids=input_ids, pixel_values=encoding.pixel_values)
752:         ...         mlm_logits = outputs.logits[0]  # shape (seq_len, vocab_size)
753:         ...         # only take into account text features (minus CLS and SEP token)
754:         ...         mlm_logits = mlm_logits[1 : input_ids.shape[1] - 1, :]
755:         ...         mlm_values, mlm_ids = mlm_logits.softmax(dim=-1).max(dim=-1)
756:         ...         # only take into account text
757:         ...         mlm_values[torch.tensor(encoded) != 103] = 0
758:         ...         select = mlm_values.argmax().item()
759:         ...         encoded[select] = mlm_ids[select].item()
760:         ...         inferred_token = [processor.decode(encoded)]
761: 
762:         >>> selected_token = ""
763:         >>> encoded = processor.tokenizer(inferred_token)
764:         >>> output = processor.decode(encoded.input_ids[0], skip_special_tokens=True)
765:         >>> print(output)
766:         a bunch of cats laying on a couch.
767:         ```"""
768:         return_dict = return_dict if return_dict is not None else self.config.return_dict
769: 
770:         outputs = self.vilt(
771:             input_ids,
772:             attention_mask=attention_mask,
773:             token_type_ids=token_type_ids,
774:             pixel_values=pixel_values,
775:             pixel_mask=pixel_mask,
776:             inputs_embeds=inputs_embeds,
777:             image_embeds=image_embeds,
778:             output_attentions=output_attentions,
779:             output_hidden_states=output_hidden_states,
780:             return_dict=return_dict,
781:         )
782: 
783:         sequence_output, pooled_output = outputs[:2]
784:         # split up final hidden states into text and image features
785:         text_seq_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
786:         text_features, _ = (sequence_output[:, :text_seq_len], sequence_output[:, text_seq_len:])
787: 
788:         mlm_logits = self.mlm_score(text_features)
789: 
790:         masked_lm_loss = None
791:         if labels is not None:
792:             loss_fct = CrossEntropyLoss()  # -100 index = padding token
793:             # move labels to correct device to enable PP
794:             labels = labels.to(mlm_logits.device)
795:             masked_lm_loss = loss_fct(mlm_logits.view(-1, self.config.vocab_size), labels.view(-1))
796: 
797:         if not return_dict:
798:             output = (mlm_logits,) + outputs[2:]
799:             return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
800: 
801:         return MaskedLMOutput(
802:             loss=masked_lm_loss,
803:             logits=mlm_logits,
804:             hidden_states=outputs.hidden_states,
805:             attentions=outputs.attentions,
806:         )
807: 
808: 
809: class ViltPredictionHeadTransform(nn.Module):
810:     def __init__(self, config):
811:         super().__init__()
812:         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
813:         if isinstance(config.hidden_act, str):
814:             self.transform_act_fn = ACT2FN[config.hidden_act]
815:         else:
816:             self.transform_act_fn = config.hidden_act
817:         self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
818: 
819:     def forward(self, hidden_states):
820:         hidden_states = self.dense(hidden_states)
821:         hidden_states = self.transform_act_fn(hidden_states)
822:         hidden_states = self.LayerNorm(hidden_states)
823:         return hidden_states
824: 
825: 
826: class ViltMLMHead(nn.Module):
827:     def __init__(self, config):
828:         super().__init__()
829:         self.config = config
830:         self.transform = ViltPredictionHeadTransform(config)
831:         self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
832: 
833:     def forward(self, x):
834:         x = self.transform(x)
835:         x = self.decoder(x)
836:         return x
837: 
838: 
839: @auto_docstring(
840:     custom_intro="""

### vqa_head (modeling_vilt.py:845-956)
845: class ViltForQuestionAnswering(ViltPreTrainedModel):
846:     def __init__(self, config):
847:         super().__init__(config)
848: 
849:         self.num_labels = config.num_labels
850:         self.vilt = ViltModel(config)
851: 
852:         # Classifier head
853:         self.classifier = nn.Sequential(
854:             nn.Linear(config.hidden_size, config.hidden_size * 2),
855:             nn.LayerNorm(config.hidden_size * 2),
856:             nn.GELU(),
857:             nn.Linear(config.hidden_size * 2, config.num_labels),
858:         )
859: 
860:         # Initialize weights and apply final processing
861:         self.post_init()
862: 
863:     @auto_docstring
864:     def forward(
865:         self,
866:         input_ids: torch.LongTensor | None = None,
867:         attention_mask: torch.FloatTensor | None = None,
868:         token_type_ids: torch.LongTensor | None = None,
869:         pixel_values: torch.FloatTensor | None = None,
870:         pixel_mask: torch.LongTensor | None = None,
871:         inputs_embeds: torch.FloatTensor | None = None,
872:         image_embeds: torch.FloatTensor | None = None,
873:         labels: torch.LongTensor | None = None,
874:         output_attentions: bool | None = None,
875:         output_hidden_states: bool | None = None,
876:         return_dict: bool | None = None,
877:         **kwargs,
878:     ) -> SequenceClassifierOutput | tuple[torch.FloatTensor]:
879:         r"""
880:         image_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`, *optional*):
881:             Optionally, instead of passing `pixel_values`, you can choose to directly pass an embedded representation.
882:             This is useful if you want more control over how to convert `pixel_values` into patch embeddings.
883:         labels (`torch.FloatTensor` of shape `(batch_size, num_labels)`, *optional*):
884:             Labels for computing the visual question answering loss. This tensor must be either a one-hot encoding of
885:             all answers that are applicable for a given example in the batch, or a soft encoding indicating which
886:             answers are applicable, where 1.0 is the highest score.
887: 
888:         Examples:
889: 
890:         ```python
891:         >>> from transformers import ViltProcessor, ViltForQuestionAnswering
892:         >>> import httpx
893:         >>> from io import BytesIO
894:         >>> from PIL import Image
895: 
896:         >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
897:         >>> with httpx.stream("GET", url) as response:
898:         ...     image = Image.open(BytesIO(response.read()))
899:         >>> text = "How many cats are there?"
900: 
901:         >>> processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
902:         >>> model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
903: 
904:         >>> # prepare inputs
905:         >>> encoding = processor(image, text, return_tensors="pt")
906: 
907:         >>> # forward pass
908:         >>> outputs = model(**encoding)
909:         >>> logits = outputs.logits
910:         >>> idx = logits.argmax(-1).item()
911:         >>> print("Predicted answer:", model.config.id2label[idx])
912:         Predicted answer: 2
913:         ```"""
914:         return_dict = return_dict if return_dict is not None else self.config.return_dict
915: 
916:         outputs = self.vilt(
917:             input_ids,
918:             attention_mask=attention_mask,
919:             token_type_ids=token_type_ids,
920:             pixel_values=pixel_values,
921:             pixel_mask=pixel_mask,
922:             inputs_embeds=inputs_embeds,
923:             image_embeds=image_embeds,
924:             output_attentions=output_attentions,
925:             output_hidden_states=output_hidden_states,
926:             return_dict=return_dict,
927:         )
928: 
929:         pooler_output = outputs.pooler_output if return_dict else outputs[1]
930: 
931:         logits = self.classifier(pooler_output)
932: 
933:         loss = None
934:         if labels is not None:
935:             # move labels to correct device to enable PP
936:             labels = labels.to(logits.device)
937:             loss = nn.functional.binary_cross_entropy_with_logits(logits, labels) * labels.shape[1]
938:             # see https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19
939: 
940:         if not return_dict:
941:             output = (logits,) + outputs[2:]
942:             return ((loss,) + output) if loss is not None else output
943: 
944:         return SequenceClassifierOutput(
945:             loss=loss,
946:             logits=logits,
947:             hidden_states=outputs.hidden_states,
948:             attentions=outputs.attentions,
949:         )
950: 
951: 
952: @auto_docstring(
953:     custom_intro="""
954:     Vilt Model transformer with a classifier head on top (a linear layer on top of the final hidden state of the [CLS]
955:     token) for image-to-text or text-to-image retrieval, e.g. MSCOCO and F30K.
956:     """

### retrieval_head (modeling_vilt.py:958-1055)
958: class ViltForImageAndTextRetrieval(ViltPreTrainedModel):
959:     def __init__(self, config):
960:         super().__init__(config)
961: 
962:         self.vilt = ViltModel(config)
963: 
964:         # Classifier head
965:         self.rank_output = nn.Linear(config.hidden_size, 1)
966: 
967:         # Initialize weights and apply final processing
968:         self.post_init()
969: 
970:     @auto_docstring
971:     def forward(
972:         self,
973:         input_ids: torch.LongTensor | None = None,
974:         attention_mask: torch.FloatTensor | None = None,
975:         token_type_ids: torch.LongTensor | None = None,
976:         pixel_values: torch.FloatTensor | None = None,
977:         pixel_mask: torch.LongTensor | None = None,
978:         inputs_embeds: torch.FloatTensor | None = None,
979:         image_embeds: torch.FloatTensor | None = None,
980:         labels: torch.LongTensor | None = None,
981:         output_attentions: bool | None = None,
982:         output_hidden_states: bool | None = None,
983:         return_dict: bool | None = None,
984:         **kwargs,
985:     ) -> SequenceClassifierOutput | tuple[torch.FloatTensor]:
986:         r"""
987:         image_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`, *optional*):
988:             Optionally, instead of passing `pixel_values`, you can choose to directly pass an embedded representation.
989:             This is useful if you want more control over how to convert `pixel_values` into patch embeddings.
990:         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
991:             Labels are currently not supported.
992: 
993:         Examples:
994: 
995:         ```python
996:         >>> from transformers import ViltProcessor, ViltForImageAndTextRetrieval
997:         >>> import httpx
998:         >>> from io import BytesIO
999:         >>> from PIL import Image
1000: 
1001:         >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
1002:         >>> with httpx.stream("GET", url) as response:
1003:         ...     image = Image.open(BytesIO(response.read()))
1004:         >>> texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]
1005: 
1006:         >>> processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")
1007:         >>> model = ViltForImageAndTextRetrieval.from_pretrained("dandelin/vilt-b32-finetuned-coco")
1008: 
1009:         >>> # forward pass
1010:         >>> scores = dict()
1011:         >>> for text in texts:
1012:         ...     # prepare inputs
1013:         ...     encoding = processor(image, text, return_tensors="pt")
1014:         ...     outputs = model(**encoding)
1015:         ...     scores[text] = outputs.logits[0, :].item()
1016:         ```"""
1017:         return_dict = return_dict if return_dict is not None else self.config.return_dict
1018: 
1019:         loss = None
1020:         if labels is not None:
1021:             raise NotImplementedError("Training is not yet supported.")
1022: 
1023:         outputs = self.vilt(
1024:             input_ids,
1025:             attention_mask=attention_mask,
1026:             token_type_ids=token_type_ids,
1027:             pixel_values=pixel_values,
1028:             pixel_mask=pixel_mask,
1029:             inputs_embeds=inputs_embeds,
1030:             image_embeds=image_embeds,
1031:             output_attentions=output_attentions,
1032:             output_hidden_states=output_hidden_states,
1033:             return_dict=return_dict,
1034:         )
1035: 
1036:         pooler_output = outputs.pooler_output if return_dict else outputs[1]
1037: 
1038:         logits = self.rank_output(pooler_output)
1039: 
1040:         if not return_dict:
1041:             output = (logits,) + outputs[2:]
1042:             return ((loss,) + output) if loss is not None else output
1043: 
1044:         return SequenceClassifierOutput(
1045:             loss=loss,
1046:             logits=logits,
1047:             hidden_states=outputs.hidden_states,
1048:             attentions=outputs.attentions,
1049:         )
1050: 
1051: 
1052: @auto_docstring(
1053:     custom_intro="""
1054:     Vilt Model transformer with a classifier head on top for natural language visual reasoning, e.g. NLVR2.
1055:     """

### nlvr_head (modeling_vilt.py:1057-1197)
1057: class ViltForImagesAndTextClassification(ViltPreTrainedModel):
1058:     def __init__(self, config):
1059:         super().__init__(config)
1060: 
1061:         self.num_labels = config.num_labels
1062:         self.vilt = ViltModel(config)
1063: 
1064:         # Classifier head
1065:         num_images = config.num_images
1066:         self.classifier = nn.Sequential(
1067:             nn.Linear(config.hidden_size * num_images, config.hidden_size * num_images),
1068:             nn.LayerNorm(config.hidden_size * num_images),
1069:             nn.GELU(),
1070:             nn.Linear(config.hidden_size * num_images, config.num_labels),
1071:         )
1072: 
1073:         # Initialize weights and apply final processing
1074:         self.post_init()
1075: 
1076:     @auto_docstring
1077:     def forward(
1078:         self,
1079:         input_ids: torch.LongTensor | None = None,
1080:         attention_mask: torch.FloatTensor | None = None,
1081:         token_type_ids: torch.LongTensor | None = None,
1082:         pixel_values: torch.FloatTensor | None = None,
1083:         pixel_mask: torch.LongTensor | None = None,
1084:         inputs_embeds: torch.FloatTensor | None = None,
1085:         image_embeds: torch.FloatTensor | None = None,
1086:         labels: torch.LongTensor | None = None,
1087:         output_attentions: bool | None = None,
1088:         output_hidden_states: bool | None = None,
1089:         return_dict: bool | None = None,
1090:         **kwargs,
1091:     ) -> ViltForImagesAndTextClassificationOutput | tuple[torch.FloatTensor]:
1092:         r"""
1093:         image_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`, *optional*):
1094:             Optionally, instead of passing `pixel_values`, you can choose to directly pass an embedded representation.
1095:             This is useful if you want more control over how to convert `pixel_values` into patch embeddings.
1096:         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
1097:             Binary classification labels.
1098: 
1099:         Examples:
1100: 
1101:         ```python
1102:         >>> from transformers import ViltProcessor, ViltForImagesAndTextClassification
1103:         >>> import httpx
1104:         >>> from io import BytesIO
1105:         >>> from PIL import Image
1106: 
1107:         >>> url_1 = "https://lil.nlp.cornell.edu/nlvr/exs/ex0_0.jpg"
1108:         >>> with httpx.stream("GET", url_1) as response:
1109:         ...     image_1 = Image.open(BytesIO(response.read()))
1110: 
1111:         >>> url_2 = "https://lil.nlp.cornell.edu/nlvr/exs/ex0_1.jpg"
1112:         >>> with httpx.stream("GET", url_2) as response:
1113:         ...     image_2 = Image.open(BytesIO(response.read()))
1114: 
1115:         >>> text = "The left image contains twice the number of dogs as the right image."
1116: 
1117:         >>> processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")
1118:         >>> model = ViltForImagesAndTextClassification.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")
1119: 
1120:         >>> # prepare inputs
1121:         >>> encoding = processor([image_1, image_2], text, return_tensors="pt")
1122: 
1123:         >>> # forward pass
1124:         >>> outputs = model(input_ids=encoding.input_ids, pixel_values=encoding.pixel_values.unsqueeze(0))
1125:         >>> logits = outputs.logits
1126:         >>> idx = logits.argmax(-1).item()
1127:         >>> print("Predicted answer:", model.config.id2label[idx])
1128:         Predicted answer: True
1129:         ```"""
1130:         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
1131:         output_hidden_states = (
1132:             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
1133:         )
1134:         return_dict = return_dict if return_dict is not None else self.config.return_dict
1135: 
1136:         if pixel_values is not None and pixel_values.ndim == 4:
1137:             # add dummy num_images dimension
1138:             pixel_values = pixel_values.unsqueeze(1)
1139: 
1140:         if image_embeds is not None and image_embeds.ndim == 3:
1141:             # add dummy num_images dimension
1142:             image_embeds = image_embeds.unsqueeze(1)
1143: 
1144:         num_images = pixel_values.shape[1] if pixel_values is not None else None
1145:         if num_images is None:
1146:             num_images = image_embeds.shape[1] if image_embeds is not None else None
1147:         if num_images != self.config.num_images:
1148:             raise ValueError(
1149:                 "Make sure to match the number of images in the model with the number of images in the input."
1150:             )
1151:         pooler_outputs = []
1152:         hidden_states = [] if output_hidden_states else None
1153:         attentions = [] if output_attentions else None
1154:         for i in range(num_images):
1155:             # forward every image through the model
1156:             outputs = self.vilt(
1157:                 input_ids,
1158:                 attention_mask=attention_mask,
1159:                 token_type_ids=token_type_ids,
1160:                 pixel_values=pixel_values[:, i, :, :, :] if pixel_values is not None else None,
1161:                 pixel_mask=pixel_mask[:, i, :, :] if pixel_mask is not None else None,
1162:                 inputs_embeds=inputs_embeds,
1163:                 image_embeds=image_embeds[:, i, :, :] if image_embeds is not None else None,
1164:                 image_token_type_idx=i + 1,
1165:                 output_attentions=output_attentions,
1166:                 output_hidden_states=output_hidden_states,
1167:                 return_dict=return_dict,
1168:             )
1169:             pooler_output = outputs.pooler_output if return_dict else outputs[1]
1170:             pooler_outputs.append(pooler_output)
1171:             if output_hidden_states:
1172:                 hidden_states.append(outputs.hidden_states)
1173:             if output_attentions:
1174:                 attentions.append(outputs.attentions)
1175: 
1176:         pooled_output = torch.cat(pooler_outputs, dim=-1)
1177:         logits = self.classifier(pooled_output)
1178: 
1179:         loss = None
1180:         if labels is not None:
1181:             loss_fct = CrossEntropyLoss()
1182:             # move labels to correct device to enable PP
1183:             labels = labels.to(logits.device)
1184:             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
1185: 
1186:         if not return_dict:
1187:             output = (logits, hidden_states, attentions)
1188:             return ((loss,) + output) if loss is not None else output
1189: 
1190:         return ViltForImagesAndTextClassificationOutput(
1191:             loss=loss,
1192:             logits=logits,
1193:             hidden_states=hidden_states,
1194:             attentions=attentions,
1195:         )
1196: 
1197: 

### token_classification_head (modeling_vilt.py:1199-1268)
1199: class ViltForTokenClassification(ViltPreTrainedModel):
1200:     def __init__(self, config):
1201:         super().__init__(config)
1202: 
1203:         self.num_labels = config.num_labels
1204:         self.vilt = ViltModel(config, add_pooling_layer=False)
1205: 
1206:         self.dropout = nn.Dropout(config.hidden_dropout_prob)
1207:         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
1208: 
1209:         # Initialize weights and apply final processing
1210:         self.post_init()
1211: 
1212:     @auto_docstring
1213:     def forward(
1214:         self,
1215:         input_ids: torch.LongTensor | None = None,
1216:         attention_mask: torch.FloatTensor | None = None,
1217:         token_type_ids: torch.LongTensor | None = None,
1218:         pixel_values: torch.FloatTensor | None = None,
1219:         pixel_mask: torch.LongTensor | None = None,
1220:         inputs_embeds: torch.FloatTensor | None = None,
1221:         image_embeds: torch.FloatTensor | None = None,
1222:         labels: torch.LongTensor | None = None,
1223:         output_attentions: bool | None = None,
1224:         output_hidden_states: bool | None = None,
1225:         return_dict: bool | None = None,
1226:         **kwargs,
1227:     ) -> TokenClassifierOutput | tuple[torch.FloatTensor]:
1228:         r"""
1229:         image_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`, *optional*):
1230:             Optionally, instead of passing `pixel_values`, you can choose to directly pass an embedded representation.
1231:             This is useful if you want more control over how to convert `pixel_values` into patch embeddings.
1232:         labels (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`, *optional*):
1233:             Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
1234:         """
1235: 
1236:         return_dict = return_dict if return_dict is not None else self.config.return_dict
1237: 
1238:         outputs = self.vilt(
1239:             input_ids,
1240:             attention_mask=attention_mask,
1241:             token_type_ids=token_type_ids,
1242:             pixel_values=pixel_values,
1243:             pixel_mask=pixel_mask,
1244:             inputs_embeds=inputs_embeds,
1245:             image_embeds=image_embeds,
1246:             output_attentions=output_attentions,
1247:             output_hidden_states=output_hidden_states,
1248:             return_dict=return_dict,
1249:         )
1250: 
1251:         sequence_output = outputs[0]
1252: 
1253:         text_input_size = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
1254: 
1255:         sequence_output = self.dropout(sequence_output)
1256:         logits = self.classifier(sequence_output[:, :text_input_size])
1257: 
1258:         loss = None
1259:         if labels is not None:
1260:             loss_fct = CrossEntropyLoss()
1261:             # move labels to correct device to enable PP
1262:             labels = labels.to(logits.device)
1263:             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
1264: 
1265:         if not return_dict:
1266:             output = (logits,) + outputs[2:]
1267:             return ((loss,) + output) if loss is not None else output
1268: 
