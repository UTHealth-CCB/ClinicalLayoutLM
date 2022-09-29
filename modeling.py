'''
The script is modified based on
https://github.com/microsoft/unilm/blob/master/layoutlmv3/layoutlmft/models/layoutlmv3/modeling_layoutlmv3.py
https://github.com/microsoft/unilm/blob/master/beit/modeling_pretrain.py
'''


import torch
import torch.nn as nn
from layoutlmft.models.layoutlmv3.modeling_layoutlmv3 import LayoutLMv3Config, \
    LayoutLMv3PreTrainedModel, \
    LayoutLMv3ClassificationHead, LayoutLMv3Model, LayoutLMv3Embeddings, LayoutLMv3Encoder, PatchEmbed
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, TokenClassifierOutput, \
    MaskedLMOutput

from timm.models.layers import trunc_normal_ as __call_trunc_normal_


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class LayoutLMv3ModelWithMask(LayoutLMv3PreTrainedModel):
    """
    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, detection=False, out_features=None, image_only=False):
        super().__init__(config)
        self.config = config
        assert not config.is_decoder and not config.add_cross_attention, \
            "This version do not support decoder. Please refer to RoBERTa for implementation of is_decoder."
        self.detection = detection
        if not self.detection:
            self.image_only = False
        else:
            assert config.visual_embed
            self.image_only = image_only

        if not self.image_only:
            self.embeddings = LayoutLMv3Embeddings(config)
        self.encoder = LayoutLMv3Encoder(config, detection=detection, out_features=out_features)

        if config.visual_embed:
            embed_dim = self.config.hidden_size
            # use the default pre-training parameters for fine-tuning (e.g., input_size)
            # when the input_size is larger in fine-tuning, we will interpolate the position embedding in forward
            self.patch_embed = PatchEmbed(embed_dim=embed_dim)

            patch_size = 16
            size = int(self.config.input_size / patch_size)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

            self.pos_embed = nn.Parameter(torch.zeros(1, size * size + 1, embed_dim))
            self.pos_drop = nn.Dropout(p=0.)

            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

            if self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
                self._init_visual_bbox(img_size=(size, size))

            from functools import partial
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            self.norm = norm_layer(embed_dim)
            trunc_normal_(self.mask_token, std=0.02)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _init_visual_bbox(self, img_size=(14, 14), max_len=1000):
        visual_bbox_x = torch.div(torch.arange(0, max_len * (img_size[1] + 1), max_len),
                                  img_size[1], rounding_mode='trunc')
        visual_bbox_y = torch.div(torch.arange(0, max_len * (img_size[0] + 1), max_len),
                                  img_size[0], rounding_mode='trunc')
        visual_bbox = torch.stack(
            [
                visual_bbox_x[:-1].repeat(img_size[0], 1),
                visual_bbox_y[:-1].repeat(img_size[1], 1).transpose(0, 1),
                visual_bbox_x[1:].repeat(img_size[0], 1),
                visual_bbox_y[1:].repeat(img_size[1], 1).transpose(0, 1),
            ],
            dim=-1,
        ).view(-1, 4)

        cls_token_box = torch.tensor([[0 + 1, 0 + 1, max_len - 1, max_len - 1]])
        self.visual_bbox = torch.cat([cls_token_box, visual_bbox], dim=0)

    def _calc_visual_bbox(self, device, dtype, bsz):  # , img_size=(14, 14), max_len=1000):
        visual_bbox = self.visual_bbox.repeat(bsz, 1, 1)
        visual_bbox = visual_bbox.to(device).type(dtype)
        return visual_bbox

    def forward_image(self, x, mask_patch):
        if self.detection:
            x = self.patch_embed(x, self.pos_embed[:, 1:, :] if self.pos_embed is not None else None)
        else:
            x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        mask_token = self.mask_token.expand(batch_size, seq_len, -1)
        # replace the masked visual tokens by mask_token
        w = mask_patch.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w


        if self.pos_embed is not None and self.detection:
            cls_tokens = cls_tokens + self.pos_embed[:, :1, :]

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None and not self.detection:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.norm(x)
        return x

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        valid_span=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        images=None,
        mask_patch=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        use_cache = False

        # if input_ids is not None and inputs_embeds is not None:
        #     raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
        elif images is not None:
            batch_size = len(images)
            device = images.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds or images")

        if not self.image_only:
            # past_key_values_length
            past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

            if attention_mask is None:
                attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if not self.image_only:
            if bbox is None:
                bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.long, device=device)

            embedding_output = self.embeddings(
                input_ids=input_ids,
                bbox=bbox,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )

        final_bbox = final_position_ids = None
        Hp = Wp = None
        if images is not None:
            patch_size = 16
            Hp, Wp = int(images.shape[2] / patch_size), int(images.shape[3] / patch_size)
            visual_emb = self.forward_image(images,mask_patch)
            if self.detection:
                visual_attention_mask = torch.ones((batch_size, visual_emb.shape[1]), dtype=torch.long, device=device)
                if self.image_only:
                    attention_mask = visual_attention_mask
                else:
                    attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
            elif self.image_only:
                attention_mask = torch.ones((batch_size, visual_emb.shape[1]), dtype=torch.long, device=device)

            if self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
                if self.config.has_spatial_attention_bias:
                    visual_bbox = self._calc_visual_bbox(device, dtype=torch.long, bsz=batch_size)
                    if self.image_only:
                        final_bbox = visual_bbox
                    else:
                        final_bbox = torch.cat([bbox, visual_bbox], dim=1)

                visual_position_ids = torch.arange(0, visual_emb.shape[1], dtype=torch.long, device=device).repeat(
                    batch_size, 1)
                if self.image_only:
                    final_position_ids = visual_position_ids
                else:
                    position_ids = torch.arange(0, input_shape[1], device=device).unsqueeze(0)
                    position_ids = position_ids.expand_as(input_ids)
                    final_position_ids = torch.cat([position_ids, visual_position_ids], dim=1)

            if self.image_only:
                embedding_output = visual_emb
            else:
                embedding_output = torch.cat([embedding_output, visual_emb], dim=1)
            embedding_output = self.LayerNorm(embedding_output)
            embedding_output = self.dropout(embedding_output)
        elif self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
            if self.config.has_spatial_attention_bias:
                final_bbox = bbox
            if self.config.has_relative_attention_bias:
                position_ids = self.embeddings.position_ids[:, :input_shape[1]]
                position_ids = position_ids.expand_as(input_ids)
                final_position_ids = position_ids

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, None, device)

        encoder_outputs = self.encoder(
            embedding_output,
            bbox=final_bbox,
            position_ids=final_position_ids,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            Hp=Hp,
            Wp=Wp,
            valid_span=valid_span,
        )

        if self.detection:
            return encoder_outputs

        sequence_output = encoder_outputs[0]
        pooled_output = None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class MLMClassifier(nn.Module):
    """
    Head for sentence-level classification tasks.
    Reference: RobertaClassificationHead
    """

    def __init__(self, dropout_rate, in_size,out_size):
        super().__init__()
        self.dense = nn.Linear(in_size, in_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(in_size, out_size)

    def forward(self, x):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class LayoutLMv3ForPretraining(LayoutLMv3PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        self.vocab_size=config.vocab_size
        self.wpa_size=config.wpa_size
        self.image_vocab_size=config.image_vocab_size

        self.layoutlmv3 = LayoutLMv3ModelWithMask(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier_mlm=nn.Linear(config.hidden_size, config.vocab_size)
        self.classifier_mim = nn.Linear(config.hidden_size, config.image_vocab_size)
        self.classifier_wpa=MLMClassifier(config.hidden_dropout_prob,config.hidden_size, config.wpa_size)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        valid_span=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        images=None,
        labels_mlm=None,
        labels_mim=None,
        labels_wpa=None,
        mask_patch=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv3(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images,
            valid_span=valid_span,
            mask_patch=mask_patch,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits_mlm = self.classifier_mlm(sequence_output)
        logits_mim=self.classifier_mim(sequence_output)
        logits_wpa=self.classifier_wpa(sequence_output)

        loss_fct = CrossEntropyLoss()
        active_logits_mlm=logits_mlm.view(-1,self.vocab_size)
        active_logits_mim = logits_mim.view(-1, self.image_vocab_size)
        active_logits_wpa = logits_wpa.view(-1, self.wpa_size)
        loss_mlm=loss_fct(active_logits_mlm,labels_mlm)
        loss_mim=loss_fct(active_logits_mim,labels_mim)
        loss_wpa=loss_fct(active_logits_wpa,labels_wpa)
        loss=loss_mlm+loss_mim+loss_wpa
        predictions_mlm=torch.argmax(logits_mlm,dim=2)
        predictions_mim = torch.argmax(logits_mim, dim=2)
        predictions_wpa = torch.argmax(logits_wpa, dim=2)
        predictions=torch.cat([predictions_mlm,predictions_mim,predictions_wpa],dim=1)

        return MaskedLMOutput(
            loss=loss,
            logits=predictions,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
