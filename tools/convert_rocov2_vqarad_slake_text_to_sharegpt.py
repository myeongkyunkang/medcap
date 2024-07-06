import json
import os
import random

from utils import MRI_TOKENS, INSTRUCTION_LIST

random.seed(0)

if __name__ == '__main__':
    print('Download ROCOv2-VQARAD-SLAKE-text in advance.')

    data_dir = './datasets/ROCOv2-VQARAD-SLAKE-text/'
    image_dir = os.path.join(data_dir, 'images_preprocess')
    train_json_path = os.path.join(data_dir, 'train.json')

    save_dir = './datasets/medcap/'
    save_train_path = os.path.join(save_dir, 'rocov2_vqarad_slake_train.json')

    # make save directory
    os.makedirs(save_dir, exist_ok=True)

    with open(train_json_path, 'r', encoding='utf-8') as f:
        train_data = json.loads(f.read())


    def convert(data, random_instruction=False):
        out = []
        for elem in data:
            out.append({
                'conversations': [
                    {"from": "human", "value": MRI_TOKENS + '\n\n' + (random.choice(INSTRUCTION_LIST) if random_instruction else INSTRUCTION_LIST[0])},
                    {"from": "gpt", "value": elem['caption'].replace('①', '(1)')}  # ① is our special token
                ],
                'image': elem['id'],
                'meta': f'dir={image_dir}'
            })
        return out


    with open(save_train_path, 'w') as json_file:
        json.dump(convert(train_data, random_instruction=True), json_file, indent=4)
