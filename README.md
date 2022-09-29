# ClinicalLayoutLM is a pre-trained multi-modal model for clinical document understanding
ClinicalLayoutLM is pre-trained based on [layoutlmv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3) 
using a corpus of ~0.35 million clincal documents.

## Example

### Document classification
```shell
your_dataset_name='your_dataset_name'
model_path='/model/path'
output_path='/output/path'
python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port 14498 run_dc.py \
  --dataset_name ${your_dataset_name} \
  --do_train --do_eval --do_predict\
  --model_name_or_path ${model_path} \
  --output_dir ${output_path} \
  --visual_embed 1 --input_size 224 \
  --max_steps 5000 \
  --save_steps 50 --evaluation_strategy steps --eval_steps 50 --logging_steps 50 \
  --learning_rate 1e-5 --per_device_train_batch_size 8 --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 16 \
  --dataloader_num_workers 2 \
  --overwrite_output_dir
```

### Pretraining
```shell
your_dataset_name='your_dataset_name'
model_path='/model/path'
output_path='/output/path'
python -m torch.distributed.launch \
  --nproc_per_node=${gpu_num} --master_port 14498 run_pretraining.py \
  --dataset_name ${your_dataset_name} \
  --do_train --do_eval --do_predict\
  --model_name_or_path microsoft/layoutlmv3-large \
  --output_dir ${output_path}/ \
  --visual_embed 1 --input_size 224 \
  --max_steps 7000 \
  --save_steps 500 --evaluation_strategy steps --eval_steps 100 --logging_steps 100\
  --learning_rate 5e-5 --per_device_train_batch_size 8 --gradient_accumulation_steps 64 \
  --per_device_eval_batch_size 16 \
  --dataloader_num_workers 4 \
  --overwrite_output_dir \
  --remove_unused_columns False
```

## Acknowledgement
The codes are based on the [transformers](https://github.com/huggingface/transformers),
, [layoutlmv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3) and  [DiT](https://github.com/microsoft/unilm/tree/master/dit) projects.

## License

The content of this project itself is licensed under the [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
