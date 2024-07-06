import json
import os
import random

from utils import MRI_TOKENS, INSTRUCTION_LIST

random.seed(0)

if __name__ == '__main__':
    print('Download VQA-RAD and VQA-RAD-text in advance.')

    vqarad_dir = './datasets/VQA_RAD/'
    image_dir = os.path.join(vqarad_dir, 'VQA_RAD Image Folder')
    train_json_path = os.path.join(vqarad_dir, 'train_text.json')
    test_json_path = os.path.join(vqarad_dir, 'test_text.json')

    save_dir = './datasets/medcap/'
    save_train_path = os.path.join(save_dir, 'vqarad_train.json')
    save_test_path = os.path.join(save_dir, 'vqarad_test.json')

    # make save directory
    os.makedirs(save_dir, exist_ok=True)

    with open(train_json_path, 'r', encoding='utf-8') as f:
        train_data = json.loads(f.read())

    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_data = json.loads(f.read())


    def convert(data, random_instruction=False):
        out = []
        for elem in data:
            out.append({
                'conversations': [
                    {"from": "human", "value": MRI_TOKENS + '\n\n' + (random.choice(INSTRUCTION_LIST) if random_instruction else INSTRUCTION_LIST[0])},
                    {"from": "gpt", "value": elem['text'].replace('①', '(1)')}  # ① is our special token
                ],
                'image': elem['image_name'],
                'meta': f'dir={image_dir}'
            })
        return out


    with open(save_train_path, 'w') as json_file:
        json.dump(convert(train_data, random_instruction=True), json_file, indent=4)

    with open(save_test_path, 'w') as json_file:
        json.dump(convert(test_data), json_file, indent=4)
