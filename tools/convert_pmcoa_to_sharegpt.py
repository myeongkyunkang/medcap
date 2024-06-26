import json
import os
import random

from utils import MRI_TOKENS, INSTRUCTION_LIST

random.seed(0)


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_object = json.loads(line)
            data.append(json_object)
    return data


if __name__ == '__main__':
    print('Download PMC-OA in advance.')

    pmcoa_dir = './datasets/pmc_oa/'
    image_dir = os.path.join(pmcoa_dir, 'caption_T060_filtered_top4_sep_v0_subfigures')
    train_jsonl_path = os.path.join(pmcoa_dir, 'train.jsonl')
    val_jsonl_path = os.path.join(pmcoa_dir, 'valid.jsonl')
    test_jsoln_path = os.path.join(pmcoa_dir, 'test.jsonl')

    save_dir = './datasets/medcap/'
    save_train_path = os.path.join(save_dir, 'pmcoa_train.json')
    save_val_path = os.path.join(save_dir, 'pmcoa_val.json')
    save_test_path = os.path.join(save_dir, 'pmcoa_test.json')

    # make save directory
    os.makedirs(save_dir, exist_ok=True)

    train_data = read_jsonl(train_jsonl_path)
    val_data = read_jsonl(val_jsonl_path)
    test_data = read_jsonl(test_jsoln_path)


    def convert(data):
        out = []
        for elem in data:
            out.append({
                'conversations': [
                    {"from": "human", "value": MRI_TOKENS + '\n\n' + random.choice(INSTRUCTION_LIST)},
                    {"from": "gpt", "value": elem['caption'].replace('①', '(1)')}  # ① is our special token
                ],
                'image': elem['image'],
                'meta': f'dir={image_dir}'
            })
        return out


    with open(save_train_path, 'w') as json_file:
        json.dump(convert(train_data), json_file, indent=4)

    with open(save_val_path, 'w') as json_file:
        json.dump(convert(val_data), json_file, indent=4)

    with open(save_test_path, 'w') as json_file:
        json.dump(convert(test_data), json_file, indent=4)
