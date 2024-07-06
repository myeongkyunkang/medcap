# Fine-tuning recipes

## PMC-OA

Download the [PMC-OA](https://huggingface.co/datasets/axiong/pmc_oa) dataset to your local machine and then run the following.

```
# preprocess PMC-OA
python tools/convert_pmcoa_to_sharegpt.py
```

### Running a fine-tuning recipe

```
python
import os
GPU=0
name='pmcoa'
vision='biomedclip'
epochs=10
data_dir='./datasets/medcap'
llama_dir='./models/llama3/Meta-Llama-3-8B-Instruct'
result_dir='./results_medcap'
postfix_result=f''
cmd=f"PYTHONPATH=./ CUDA_VISIBLE_DEVICES={GPU} python torchtune/_cli/tune.py run full_finetune_single_device --config llama3/8B_full_single_device \
    seed=1 \
    epochs={epochs} \
    batch_size=16 \
    gradient_accumulation_steps=1 \
    vision={vision} \
    dataset._component_=torchtune.datasets.chat_dataset \
    dataset.source={data_dir}/{name}_train.json \
    dataset.conversation_style=sharegpt \
    dataset.chat_format=ChatFormat \
    dataset.max_seq_len=400 \
    tokenizer.path={llama_dir}/tokenizer.model \
    checkpointer.checkpoint_dir={llama_dir} \
    checkpointer.output_dir={result_dir}/{name}_{vision}{postfix_result}/checkpoint \
    metric_logger.log_dir={result_dir}/{name}_{vision}{postfix_result}/log \
    output_dir={result_dir}/{name}_{vision}{postfix_result}"
print(cmd)
os.system(cmd)
```

##  ROCOv2-VQARAD-SLAKE-text

Download the [ROCOv2-VQARAD-SLAKE-text](https://huggingface.co/datasets/myeongkyunkang/ROCOv2-VQARAD-SLAKE-text) dataset to your local machine and then run the following.

```
# preprocess ROCOv2
python tools/convert_rocov2_vqarad_slake_text_to_sharegpt.py
```

### Running a fine-tuning recipe

```
python
import os
GPU=0
name='rocov2_vqarad_slake'
vision='biomedclip'
epochs=10
data_dir='./datasets/medcap'
llama_dir='./models/llama3/Meta-Llama-3-8B-Instruct'
result_dir='./results_medcap'
pretrained_medcap='./models/medcap-pmcoa/last.pt'
postfix_result=f''
cmd=f"PYTHONPATH=./ CUDA_VISIBLE_DEVICES={GPU} python torchtune/_cli/tune.py run full_finetune_single_device --config llama3/8B_full_single_device \
    seed=1 \
    epochs={epochs} \
    batch_size=16 \
    gradient_accumulation_steps=1 \
    vision={vision} \
    dataset._component_=torchtune.datasets.chat_dataset \
    dataset.source={data_dir}/{name}_train.json \
    dataset.conversation_style=sharegpt \
    dataset.chat_format=ChatFormat \
    dataset.max_seq_len=400 \
    tokenizer.path={llama_dir}/tokenizer.model \
    checkpointer.checkpoint_dir={llama_dir} \
    vision_checkpoint={pretrained_medcap} \
    checkpointer.output_dir={result_dir}/{name}_{vision}{postfix_result}/checkpoint \
    metric_logger.log_dir={result_dir}/{name}_{vision}{postfix_result}/log \
    output_dir={result_dir}/{name}_{vision}{postfix_result}"
print(cmd)
os.system(cmd)
```
