# Tutorials

## VQA-RAD-text

Download the [VQA-RAD](https://osf.io/89kps/) and [VQA-RAD-text](https://huggingface.co/datasets/myeongkyunkang/VQA-RAD-text) datasets to your local machine and then run the following.
Afterward, download the vision encoder [medcap-textplus-pmcoa-patients-llama3.1](https://huggingface.co/myeongkyunkang/medcap-textplus-pmcoa-patients-llama3.1) to your local machine.

```
python tools/convert_to_sharegpt.py --dataset vqarad
```

### Running a fine-tuning recipe

```
python
import os
GPU=0
name='vqarad'
vision='biomedclip'
epochs=5
data_dir='./datasets/medcap'
llama_dir='./models/llama3/Meta-Llama-3.1-8B-Instruct'
pretrained_medcap='./models/medcap-textplus-pmcoa-patients-llama3.1/meta_model_last.pt'
result_dir='./results_medcap'
postfix_result=''
cmd=f"PYTHONPATH=./ CUDA_VISIBLE_DEVICES={GPU} python torchtune/_cli/tune.py run full_finetune_single_device --config recipes/configs/llama3_1/8B_full_single_device.yaml \
    epochs={epochs} \
    batch_size=4 \
    gradient_accumulation_steps=1 \
    optimizer.lr=2e-5 \
    vision={vision} \
    dataset._component_=torchtune.datasets.chat_dataset \
    dataset.source={data_dir}/{name}_train.json \
    dataset.conversation_style=sharegpt \
    dataset.max_seq_len=500 \
    tokenizer.path={llama_dir}/original/tokenizer.model \
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
name='vqarad'
vision='biomedclip'
epoch=5
data_dir='./datasets/medcap'
llama_dir='./models/llama3/Meta-Llama-3.1-8B-Instruct'
result_dir='./results_medcap'
postfix_result=''
vqa_type='no_cond_desc'
cmd=f"PYTHONPATH=./ CUDA_VISIBLE_DEVICES={GPU} python torchtune/_cli/tune.py run recipes/generate.py --config recipes/configs/custom_generation_config.yaml \
    test_metrics=True \
    vqa_type={vqa_type} \
    seed=1 \
    max_new_tokens=500 \
    top_k=null \
    temperature=0 \
    vision={vision} \
    dataset_source={data_dir}/{name}_test.json \
    dataset_conversation_style=sharegpt \
    dataset_max_seq_len=500 \
    tokenizer.path={llama_dir}/original/tokenizer.model \
    checkpointer.checkpoint_dir={llama_dir} \
    vision_checkpoint={result_dir}/{name}_{vision}{postfix_result}/checkpoint/meta_model_{epoch}.pt \
    output_dir={result_dir}/{name}_{vision}{postfix_result}"
print(cmd)
os.system(cmd)
```

### Modify configs for VQA-RAD

```
test_vqarad=True
dataset_source=./datasets/
```

### Running VQA evaluation using Meta-Llama-3.1-70B-Instruct

```
python
import os
GPU=0
bit=4
name='vqarad'
vision='biomedclip'
epoch=5
llama_dir='./models/llama3/Meta-Llama-3.1-70B-Instruct'
result_dir='./results_medcap'
postfix_result=''
vqa_name='vqarad'
vqa_type='no_cond_desc'
cmd=f"CUDA_VISIBLE_DEVICES={GPU} python chat_llama3_quant.py \
    --bit {bit} \
    --ckpt_dir {llama_dir} \
    --exp eval_vqa \
    --csv_path {result_dir}/{name}_{vision}{postfix_result}/{vqa_name}-{vqa_type}-raw-meta_model_{epoch}.csv"
print(cmd)
os.system(cmd)
```
