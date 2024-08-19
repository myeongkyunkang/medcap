# Fine-tuning recipes

## *-text plus datasets

Download the datasets to your local machine and then run the following.

- Download the [PMC-VQA-text](https://huggingface.co/datasets/myeongkyunkang/PMC-VQA-text) dataset to your local machine.
- Download the [ROCOv2](https://zenodo.org/records/8333645) (train_images.zip and train_captions.csv) dataset to your local machine.
- Download the [LLaVA-Med-60K-IM-text](https://huggingface.co/datasets/myeongkyunkang/LLaVA-Med-60K-IM-text) dataset to your local machine and then run "unzip images.zip -d images_med60k" and "mv train_text.json train_text_med60k.json" in advance.
- Download the [PubMedVision](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision) (images_*.zip and PubMedVision_Alignment_VQA.json) dataset to your local machine and then run "for ((i=0; i<20; i++)) do unzip -j images_$i.zip -d images_vision/ & done" in advance.
- Download the [MedICaT](https://github.com/allenai/medicat) dataset to your local machine.
- Download the [PMC-OA](https://huggingface.co/datasets/axiong/pmc_oa) dataset to your local machine.
- Download [PMC-Patients](https://huggingface.co/datasets/zhengyun21/PMC-Patients) (PMC-Patients.csv) and [PMC-Patients-images](https://huggingface.co/datasets/myeongkyunkang/PMC-Patients-images) datasets to your local machine and then run "unzip images.zip -d images_patients" in advance.

```
python tools/convert_to_sharegpt.py --dataset textplus
```

### Running a fine-tuning recipe

```
python
import os
GPU='0,1,2,3,4,5,6,7'
NUM_GPU=len(GPU.split(','))
name='textplus_pmcoa_patients'
vision='biomedclip'
epochs=5
data_dir='./datasets/medcap'
llama_dir='./models/llama3/Meta-Llama-3.1-8B-Instruct'
pretrained_medcap='./models/medcap-textplus-pmcoa-patients-llama3.1/meta_model_last.pt'
result_dir='./results_medcap'
postfix_result=''
cmd=f"PYTHONPATH=./ CUDA_VISIBLE_DEVICES={GPU} python torchtune/_cli/tune.py run --master_port 29500 --nproc_per_node {NUM_GPU} full_finetune_distributed --config recipes/configs/llama3_1/8B_full.yaml \
    epochs={epochs} \
    batch_size=8 \
    gradient_accumulation_steps=1 \
    optimizer.lr=2e-5 \
    vision={vision} \
    dataset._component_=torchtune.datasets.chat_dataset \
    dataset.source={data_dir}/{name}_train.json \
    dataset.conversation_style=sharegpt \
    dataset.max_seq_len=1000 \
    tokenizer.path={llama_dir}/original/tokenizer.model \
    checkpointer.checkpoint_dir={llama_dir} \
    vision_checkpoint={pretrained_medcap} \
    checkpointer.output_dir={result_dir}/{name}_{vision}{postfix_result}/checkpoint \
    metric_logger.log_dir={result_dir}/{name}_{vision}{postfix_result}/log \
    output_dir={result_dir}/{name}_{vision}{postfix_result}"
print(cmd)
os.system(cmd)
```
