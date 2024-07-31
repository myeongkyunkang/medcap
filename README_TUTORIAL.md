# Tutorials

## SLAKE-text

Download the [SLAKE](https://www.med-vqa.com/slake/) and [SLAKE-text](https://huggingface.co/datasets/myeongkyunkang/SLAKE-text) datasets to your local machine and then run the following.
Afterward, download the vision encoder [medcap-textplus-pmcoa-patients-llama3](https://huggingface.co/myeongkyunkang/medcap-textplus-pmcoa-patients-llama3) to your local machine.

```
python tools/convert_to_sharegpt.py --dataset slake
```

### Running a fine-tuning recipe

```
python
import os
GPU=0
name='slake'
vision='biomedclip'
epochs=5
data_dir='./datasets/medcap'
llama_dir='./models/llama3/Meta-Llama-3-8B-Instruct'
pretrained_medcap='./models/medcap-textplus-pmcoa-patients-llama3/last.pt'
result_dir='./results_medcap'
postfix_result=''
cmd=f"PYTHONPATH=./ CUDA_VISIBLE_DEVICES={GPU} python torchtune/_cli/tune.py run full_finetune_single_device --config llama3/8B_full_single_device \
    seed=1 \
    epochs={epochs} \
    batch_size=16 \
    gradient_accumulation_steps=1 \
    optimizer.lr=2e-5 \
    vision={vision} \
    dataset._component_=torchtune.datasets.chat_dataset \
    dataset.source={data_dir}/{name}_train.json \
    dataset.conversation_style=sharegpt \
    dataset.chat_format=ChatFormat \
    dataset.max_seq_len=500 \
    tokenizer.path={llama_dir}/tokenizer.model \
    checkpointer.checkpoint_dir={llama_dir} \
    vision_checkpoint={pretrained_medcap} \
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
GPU=0
name='slake'
vision='biomedclip'
epochs=5
data_dir='./datasets/medcap'
llama_dir='./models/llama3/Meta-Llama-3-8B-Instruct'
result_dir='./results_medcap'
postfix_result=''
for epoch in range(epochs):
    cmd=f"PYTHONPATH=./ CUDA_VISIBLE_DEVICES={GPU} python torchtune/_cli/tune.py run recipes/generate.py --config recipes/configs/custom_generation_config.yaml \
        test_metrics=True \
        seed=1 \
        max_new_tokens=200 \
        top_k=null \
        temperature=0 \
        vision={vision} \
        dataset_source={data_dir}/{name}_val.json \
        dataset_conversation_style=sharegpt \
        dataset_chat_format=ChatFormat \
        dataset_max_seq_len=200 \
        tokenizer.path={llama_dir}/tokenizer.model \
        checkpointer.checkpoint_dir={llama_dir} \
        vision_checkpoint={result_dir}/{name}_{vision}{postfix_result}/checkpoint/meta_model_{epoch}.pt \
        output_dir={result_dir}/{name}_{vision}{postfix_result}"
    print(cmd)
    os.system(cmd)
```

### Modify configs for SLAKE VQA

```
test_slake=True
dataset_source=./datasets/SLAKE
```

### Modify configs for VQA-RAD

```
test_vqarad=True
dataset_source=./datasets/VQA_RAD
```