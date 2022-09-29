
'''
The script is modified based on
https://github.com/microsoft/unilm/blob/master/layoutlmv3/layoutlmft/data/data_collator.py
https://github.com/microsoft/unilm/blob/master/beit2/masking_generator.py
https://github.com/facebookresearch/SpanBERT/blob/main/pretraining/fairseq/data/masking.py
'''


import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.data.data_collator import (
    DataCollatorMixin,
    _torch_collate_batch,
)
from transformers.file_utils import PaddingStrategy
import numpy as np
import math,random

def pre_calc_rel_mat(segment_ids):
    valid_span = torch.zeros((segment_ids.shape[0], segment_ids.shape[1], segment_ids.shape[1]),
                             device=segment_ids.device, dtype=torch.bool)
    for i in range(segment_ids.shape[0]):
        for j in range(segment_ids.shape[1]):
            valid_span[i, j, :] = segment_ids[i, :] == segment_ids[i, j]

    return valid_span


class MaskingGenerator:
    def __init__(
            self, input_size, num_masking_patches, min_num_patches=4, max_num_patches=None,
            min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros(shape=self.get_shape(), dtype=np.int32)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        # maintain a fix number {self.num_masking_patches}
        if mask_count > self.num_masking_patches:
            delta = mask_count - self.num_masking_patches
            mask_x, mask_y = mask.nonzero()
            to_vis = np.random.choice(mask_x.shape[0], delta, replace=False)
            mask[mask_x[to_vis], mask_y[to_vis]] = 0

        elif mask_count < self.num_masking_patches:
            delta = self.num_masking_patches - mask_count
            mask_x, mask_y = (mask == 0).nonzero()
            to_mask = np.random.choice(mask_x.shape[0], delta, replace=False)
            mask[mask_x[to_mask], mask_y[to_mask]] = 1

        assert mask.sum() == self.num_masking_patches, f"mask: {mask}, mask count {mask.sum()}"

        return mask


@dataclass
class DataCollatorForPretraining(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    image_mask_generator=None

    def __call__(self, features):

        images = None
        if "images" in features[0]:
            images = torch.stack([torch.tensor(d.pop("images")) for d in features])
            IMAGE_LEN = int(images.shape[-1] / 16) * int(images.shape[-1] / 16) + 1

        patch_tokens=torch.stack([torch.tensor(d.pop("patch_token")) for d in features])

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors=None,
        )

        if images is not None:
            batch["images"] = images
            batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) and k == 'attention_mask' else v
                     for k, v in batch.items()}
            visual_attention_mask = torch.ones((len(batch['input_ids']), IMAGE_LEN), dtype=torch.long)
            batch["attention_mask"] = torch.cat([batch['attention_mask'], visual_attention_mask], dim=1)

        has_bbox_input = "bbox" in features[0]
        has_aligned_patch="aligned_patch" in features[0]
        has_aligned_patch_token = "aligned_patch_token" in features[0]
        has_word_start_marker = "word_start_marker" in features[0]
        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        if has_bbox_input:
            batch["bbox"] = [bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox)) for bbox in batch["bbox"]]
        if has_aligned_patch:
            batch["aligned_patch"] = [x + [-100] * (sequence_length - len(x)) for x in batch["aligned_patch"]]
        if has_aligned_patch_token:
            batch["aligned_patch_token"] = [x + [-100] * (sequence_length - len(x)) for x in batch["aligned_patch_token"]]
        if has_word_start_marker:
            batch["word_start_marker"] = [x + [-100] * (sequence_length - len(x)) for x in batch["word_start_marker"]]

        batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) else v for k, v in batch.items()}
        inputs_text,labels_mlm, masked_token_indices,unmasked_token_indices=self.mask_text(batch)
        masked_patch_indices=self.mask_image(batch)
        labels_mim=masked_patch_indices*patch_tokens
        labels_mim = torch.where(labels_mim == 0, -100, labels_mim)
        tmp=torch.where(batch['aligned_patch']==-100, 0, batch['aligned_patch'])
        batch_size=tmp.shape[0]
        aligned_patch_mask_info = torch.gather(masked_patch_indices, 1,
                                               tmp)
        labels_wpa=unmasked_token_indices*(1-aligned_patch_mask_info)
        labels_wpa=labels_wpa.type(torch.int64)
        labels_wpa = torch.where(unmasked_token_indices == 0, -100, labels_wpa)
        mask_padding_len_mlm=IMAGE_LEN
        mask_padding_len_mim = labels_mlm.shape[1]+1
        mask_padding_len_wpa = IMAGE_LEN

        active_labels_mlm = torch.cat( (labels_mlm,torch.full((batch_size,mask_padding_len_mlm),-100)),dim=1)
        active_labels_mim = torch.cat( (torch.full((batch_size, mask_padding_len_mim), -100), labels_mim) , dim=1)
        active_labels_wpa = torch.cat( (labels_wpa, torch.full((batch_size, mask_padding_len_wpa), -100)), dim=1)

        batch['labels_mlm'] = active_labels_mlm.view(-1)
        batch['labels_mim'] = active_labels_mim.view(-1)
        batch['labels_wpa'] = active_labels_wpa.view(-1)
        batch['labels'] = torch.cat([active_labels_mlm, active_labels_mim, active_labels_wpa], dim=1)
        batch['mask_patch'] = masked_patch_indices

        aligned_path = batch.pop("aligned_patch")
        aligned_patch_token=batch.pop("aligned_patch_token")
        word_start_marker=batch.pop('word_start_marker')

        return batch

    def mask_image(self,batch):
        batch_size=batch['input_ids'].shape[0]
        if self.image_mask_generator is None:
            image_mask_rate=0.4
            input_size=14
            num_masking_patches=int(input_size**2*image_mask_rate)
            self.image_mask_generator=MaskingGenerator(input_size=input_size, num_masking_patches=num_masking_patches, min_num_patches=16)
        masked_image_indices=[self.image_mask_generator().flatten() for _ in range(batch_size)]
        masked_image_indices=np.vstack(masked_image_indices)
        return torch.tensor(masked_image_indices, dtype=torch.int64)
        pass

    def mask_text(self, batch):
        word_start_marker = batch['word_start_marker']
        input_ids = batch['input_ids']
        num_tokens = torch.sum(torch.where(input_ids >1 , 1, 0)).item()
        num_words = torch.sum(torch.where(word_start_marker ==1 , 1, 0)).item()
        labels_text = batch['input_ids'].clone()
        token_probability = 0.3
        poisson_rate = 3.0
        num_masked_tokens = int(num_tokens * token_probability)
        span_accept_rate = (num_masked_tokens)/num_tokens/poisson_rate
        span_accept_rate =token_probability/ poisson_rate
        batch_size, seq_len = word_start_marker.shape
        probability_matrix = torch.where(word_start_marker == 1, poisson_rate, 0.0)
        sampled_span_length = torch.poisson(probability_matrix)
        masked_indices = torch.zeros(sampled_span_length.shape)

        def is_accepted():
            res=np.random.rand()
            return res < span_accept_rate

        masked_tokens = 0
        for i in range(batch_size):
            if masked_tokens > num_masked_tokens: break
            j = 0
            while j < seq_len-1 and masked_tokens <= num_masked_tokens:
                if sampled_span_length[i, j] > 0 and word_start_marker[i, j] == 1:
                    accepted = is_accepted()
                    if not accepted:
                        j += 1
                        continue
                    span_len = sampled_span_length[i, j]
                    cur_masked_words = 0
                    while j < seq_len-1:
                        masked_indices[i, j] = 1
                        masked_tokens += 1
                        j += 1
                        if j ==seq_len-1:
                            break
                        if word_start_marker[i, j] == 1:
                            cur_masked_words += 1
                        if cur_masked_words == span_len:
                            break
                j += 1
        unmasked_token_indices=(1 - masked_indices) * torch.where(word_start_marker >= 0, 1, 0)
        masked_indices_bool = masked_indices.bool()

        labels_text[~masked_indices_bool] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels_text.shape, 0.8)).bool() & masked_indices_bool
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels_text.shape, 0.5)).bool() & masked_indices_bool & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels_text.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels_text,masked_indices,unmasked_token_indices

@dataclass
class DataCollatorForDocumentClassification(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        images = None
        if "images" in features[0]:
            images = torch.stack([torch.tensor(d.pop("images")) for d in features])
            IMAGE_LEN = int(images.shape[-1] / 16) * int(images.shape[-1] / 16) + 1

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if images is not None:
            batch["images"] = images
            batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) and k == 'attention_mask' else v
                     for k, v in batch.items()}
            visual_attention_mask = torch.ones((len(batch['input_ids']), IMAGE_LEN), dtype=torch.long)
            batch["attention_mask"] = torch.cat([batch['attention_mask'], visual_attention_mask], dim=1)

        if labels is None:
            return batch

        has_bbox_input = "bbox" in features[0]
        has_position_input = "position_ids" in features[0]
        padding_idx=self.tokenizer.pad_token_id
        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            if has_bbox_input:
                batch["bbox"] = [bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox)) for bbox in batch["bbox"]]
            if has_position_input:
                batch["position_ids"] = [position_id + [padding_idx] * (sequence_length - len(position_id))
                                          for position_id in batch["position_ids"]]

        else:
            if has_bbox_input:
                batch["bbox"] = [[[0, 0, 0, 0]] * (sequence_length - len(bbox)) + bbox for bbox in batch["bbox"]]
            if has_position_input:
                batch["position_ids"] = [[padding_idx] * (sequence_length - len(position_id))
                                          + position_id for position_id in batch["position_ids"]]

        if 'segment_ids' in batch:
            assert 'position_ids' in batch
            for i in range(len(batch['segment_ids'])):
                batch['segment_ids'][i] = batch['segment_ids'][i] + [batch['segment_ids'][i][-1] + 1] * (sequence_length - len(batch['segment_ids'][i])) + [
                    batch['segment_ids'][i][-1] + 2] * IMAGE_LEN

        batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) else v for k, v in batch.items()}
        batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
        if 'segment_ids' in batch:
            valid_span = pre_calc_rel_mat(
                segment_ids=batch['segment_ids']
            )
            batch['valid_span'] = valid_span
            del batch['segment_ids']

        if images is not None:
            visual_labels = torch.ones((len(batch['input_ids']), IMAGE_LEN), dtype=torch.long) * -100

        return batch