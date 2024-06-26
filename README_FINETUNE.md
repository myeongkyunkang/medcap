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
GPU='0'
name='pmcoa'
vision='biomedclip'
epochs=10
seed=1
data_dir='./datasets/medcap'
llama_dir='./models/llama3/Meta-Llama-3-8B-Instruct'
result_dir='./results_medcap'
postfix_result=f'_trainonly'
cmd=f"PYTHONPATH=./ CUDA_VISIBLE_DEVICES={GPU} python torchtune/_cli/tune.py run full_finetune_single_device --config llama3/8B_full_single_device \
    seed={seed} \
    epochs={epochs} \
    batch_size=16 \
    gradient_accumulation_steps=1 \
    vision={vision} \
    trainonly=True \
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

### Running a test script

```
python
import os
GPU='0'
name='pmcoa'
vision='biomedclip'
epochs=10
seed=1
data_dir='./datasets/medcap'
llama_dir='./models/llama3/Meta-Llama-3-8B-Instruct'
result_dir='./results_medcap'
postfix_result=f'_trainonly'
for epoch in range(epochs):
    cmd=f"PYTHONPATH=./ CUDA_VISIBLE_DEVICES={GPU} python torchtune/_cli/tune.py run recipes/generate.py --config recipes/configs/custom_generation_config.yaml \
        test_metrics=True \
        seed={seed} \
        max_new_tokens=200 \
        top_k=null \
        temperature=0 \
        vision={vision} \
        dataset_source={data_dir}/{name}_val.json \
        dataset_conversation_style=sharegpt \
        dataset_chat_format=ChatFormat \
        dataset_max_seq_len=200 \
        tokenizer.path={llama_dir}/tokenizer.model \
        checkpointer.checkpoint_files=[meta_model_{epoch}.pt] \
        checkpointer.checkpoint_dir={result_dir}/{name}_{vision}{postfix_result}/checkpoint/ \
        output_dir={result_dir}/{name}_{vision}{postfix_result}"
    print(cmd)
    os.system(cmd)
```

### Modify configs for SLAKE

```
test_vqaslake=True
dataset_source=./datasets/SLAKE
```

### Modify configs for VQA-RAD

```
test_vqarad=True
dataset_source=./datasets/VQA_RAD
```