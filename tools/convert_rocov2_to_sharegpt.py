import json
import os
import random

import pandas as pd

from utils import MRI_TOKENS, INSTRUCTION_LIST

random.seed(0)

if __name__ == '__main__':
    print('Download ROCOv2 in advance.')

    rocov2_dir = './datasets/ROCOv2/'
    image_dir = os.path.join(rocov2_dir, 'images')
    train_csv_path = os.path.join(rocov2_dir, 'train_captions.csv')
    val_csv_path = os.path.join(rocov2_dir, 'valid_captions.csv')
    test_csv_path = os.path.join(rocov2_dir, 'test_captions.csv')

    save_dir = './datasets/medcap/'
    save_train_path = os.path.join(save_dir, 'rocov2_train.json')
    save_val_path = os.path.join(save_dir, 'rocov2_val.json')
    save_test_path = os.path.join(save_dir, 'rocov2_test.json')

    # make save directory
    os.makedirs(save_dir, exist_ok=True)

    train_data = pd.read_csv(train_csv_path)
    val_data = pd.read_csv(val_csv_path)
    test_data = pd.read_csv(test_csv_path)


    def convert(data, image_dir, random_instruction=False):
        out = []
        for id, caption in zip(data['ID'], data['Caption']):
            out.append({
                'conversations': [
                    {"from": "human", "value": MRI_TOKENS + '\n\n' + (random.choice(INSTRUCTION_LIST) if random_instruction else INSTRUCTION_LIST[0])},
                    {"from": "gpt", "value": caption.strip().replace('①', '(1)')}  # ① is our special token
                ],
                'image': f'{id}.jpg',
                'meta': f'dir={image_dir}'
            })
        return out


    with open(save_train_path, 'w') as json_file:
        json.dump(convert(train_data, image_dir=os.path.join(image_dir, 'train'), random_instruction=True), json_file, indent=4)

    with open(save_val_path, 'w') as json_file:
        json.dump(convert(val_data, image_dir=os.path.join(image_dir, 'valid')), json_file, indent=4)

    with open(save_test_path, 'w') as json_file:
        json.dump(convert(test_data, image_dir=os.path.join(image_dir, 'test')), json_file, indent=4)
