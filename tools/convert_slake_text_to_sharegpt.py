import json
import os
import random

from utils import MRI_TOKENS, INSTRUCTION_LIST

random.seed(0)

if __name__ == '__main__':
    print('Download SLAKE and SLAKE-text in advance.')

    slake_dir = './datasets/SLAKE/'
    image_dir = os.path.join(slake_dir, 'imgs')
    train_json_path = os.path.join(slake_dir, 'train_text.json')
    val_json_path = os.path.join(slake_dir, 'validation_text.json')
    test_jsoln_path = os.path.join(slake_dir, 'test_text.json')
    num_increase = 2

    save_dir = './datasets/medcap/'
    save_train_path = os.path.join(save_dir, 'slake_train.json')
    save_val_path = os.path.join(save_dir, 'slake_val.json')
    save_test_path = os.path.join(save_dir, 'slake_test.json')

    # make save directory
    os.makedirs(save_dir, exist_ok=True)

    with open(train_json_path, 'r', encoding='utf-8') as f:
        train_data = json.loads(f.read())

    with open(val_json_path, 'r', encoding='utf-8') as f:
        val_data = json.loads(f.read())

    with open(test_jsoln_path, 'r', encoding='utf-8') as f:
        test_data = json.loads(f.read())


    def convert(data):
        out = []
        for elem in data:
            out.append({
                'conversations': [
                    {"from": "human", "value": MRI_TOKENS + '\n\n' + random.choice(INSTRUCTION_LIST)},
                    {"from": "gpt", "value": elem['text'].replace('①', '(1)')}  # ① is our special token
                ],
                'image': elem['img_name'],
                'meta': f'dir={image_dir}'
            })
        return out


    with open(save_train_path, 'w') as json_file:
        save_train_data = []
        for _ in range(num_increase):
            save_train_data.extend(convert(train_data))
        json.dump(save_train_data, json_file, indent=4)

    with open(save_val_path, 'w') as json_file:
        json.dump(convert(val_data), json_file, indent=4)

    with open(save_test_path, 'w') as json_file:
        json.dump(convert(test_data), json_file, indent=4)
