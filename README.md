# medcap: LLM-Powered Medical Image Captioning and Visual Question Answering

<img src="diagram.jpg">

## Introduction

**Why Fine-Tune Only the Vision Encoder?**
We assume that publicly available instruction-tuned LLMs (e.g., Meta-Llama-3.1-8B-Instruct) have high generalization capabilities.
Instead of struggling to prevent LLM overfitting, we freeze LLM parameters during fine-tuning and focus on enhancing the vision input.
By using vision encoders compatible with LLMs that offer high-quality representation, this approach is expected to leverage the LLMs' generalization abilities, reducing concerns about overfitting to in-house data.
Moreover, a vision encoder trained with LLMs enables high-quality feature extraction, which is expected to show superior performance in various downstream tasks.

Key features added in this repository include:

- Enabling LLMs to process image inputs.
- Training the vision encoder while keeping the LLMs frozen.

**Backgrounds.**
When using Large Language Models (LLMs) with in-house data, we often experience that LLMs do not perform as well as they do with natural images.
To address this, people construct an in-house instruction dataset and use it to fine-tune LLMs.
However, even with this fine-tuning, the models still struggle with out-of-scope inputs (i.e., overfitting), manifesting as:

- Degraded performance in **external** validation.
- Poor responses to **unexpected** questions.
- **Overfitting** to the training chat format (e.g., weak performance in visual question-answering tasks when fine-tuned with image-caption pairs).

**Design Principle.**
This repository is designed to develop image captioning and visual question answering with minimal effort.
To achieve this, we have made minimal modifications to the original torchtune codebase, ensuring easy tracking of changes.

```
Check via Ctrl+F "# UPDATED" # less than 100 lines updated
Add recipes/configs/custom_generation_config.yaml  # use "tune cp generation" and update to be compatible with llama3
Add torchtune/custom_generate_func.py  # for custom testing
Update recipes/full_finetune_distributed.py
Update recipes/full_finetune_single_device.py
Update recipes/generate.py
Update torchtune/datasets/_chat.py
Update torchtune/modules/transformer.py
Update torchtune/utils/_generation.py
Update torchtune/utils/collate.py 
```

## Preparations

Follow the instructions on the official [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) repository to ensure you have access to the official Llama model weights.
Once you have confirmed access, download the weights to your local machine.
Afterward, download the vision encoder [medcap-textplus-pmcoa-patients-llama3.1](https://huggingface.co/myeongkyunkang/medcap-textplus-pmcoa-patients-llama3.1) to your local machine.

## OmniMedVQA evaluation

### Running a test script for OmniMedVQA

Download the [OmniMedVQA](https://huggingface.co/datasets/foreverbeliever/OmniMedVQA) dataset to your local machine.

```
python
import os
GPU=0
name='textplus_pmcoa_patients'
vision='biomedclip'
epoch='last'
data_dir='./datasets/medcap'
llama_dir='./models/llama3/Meta-Llama-3.1-8B-Instruct'
result_dir='./results_medcap'
postfix_result=''
test='omnimedvqa'
vqa_type='no_cond_desc'
cmd=f"PYTHONPATH=./ CUDA_VISIBLE_DEVICES={GPU} python torchtune/_cli/tune.py run recipes/generate.py --config recipes/configs/custom_generation_config.yaml \
    test_{test}=True \
    vqa_type={vqa_type} \
    seed=1 \
    max_new_tokens=200 \
    top_k=null \
    temperature=0 \
    vision={vision} \
    dataset_source=./datasets/ \
    dataset_conversation_style=sharegpt \
    dataset_max_seq_len=500 \
    tokenizer.path={llama_dir}/original/tokenizer.model \
    checkpointer.checkpoint_dir={llama_dir} \
    vision_checkpoint={result_dir}/{name}_{vision}{postfix_result}/checkpoint/meta_model_{epoch}.pt \
    output_dir={result_dir}/{name}_{vision}{postfix_result}"
print(cmd)
os.system(cmd)
```

## Fine-tuning recipes

Please refer to [README_FINETUNE.md](README_FINETUNE.md)

## Tutorials

Please refer to [README_TUTORIAL.md](README_TUTORIAL.md)

## Pretrained models

<table><tbody>
<tr><td>*-text+ datasets</td>
<td><a href="https://huggingface.co/myeongkyunkang/medcap-textplus-pmcoa-patients-llama3.1">download</a></td></tr>
</tbody></table>

## Requirements

```
conda create -n medcap python=3.10 -y
source activate medcap

pip install pip==24.0 setuptools==69.5.1 packaging==24.0 numpy==1.26.2
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

pip install torchtune==0.2.1
pip uninstall torchtune -y
pip install bitsandbytes

pip install pandas scikit-learn
pip install "git+https://github.com/salaniz/pycocoevalcap.git"
pip install open_clip_torch==2.24.0 transformers==4.43.3

pip install accelerate
```

## Citation

If you find this repository useful in your research, please cite:

```
@misc{medcap,
    title={medcap: LLM-Powered Medical Image Captioning and Visual Question Answering},
    author={Kang, Myeongkyun},
    howpublished={\url{https://github.com/myeongkyunkang/medcap}},
    year={2024}
}
```

## Acknowledgements

Thanks to works below for their implementations which were useful for this work.
[torchtune](https://github.com/pytorch/torchtune/tree/v0.2.1),
[BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
