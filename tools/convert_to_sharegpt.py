import argparse
import json
import os
import random

import pandas as pd

from utils import *

random.seed(0)


def convert(data, image_key, image_dir, text_key='text', random_instruction=False, instruction_list=INSTRUCTION_LIST):
    out = []
    for elem in data:
        out.append({
            'conversations': [
                {"from": "human", "value": MRI_TOKENS + '\n\n' + (random.choice(instruction_list) if random_instruction else instruction_list[0])},
                {"from": "gpt", "value": elem[text_key].replace('①', '(1)')}  # ① is our special token
            ],
            'image': elem[image_key],
            'meta': f'dir={image_dir}'
        })
    return out


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_object = json.loads(line)
            data.append(json_object)
    return data


def convert_pmcoa(pmcoa_dir, save_dir):
    image_dir = os.path.join(pmcoa_dir, 'caption_T060_filtered_top4_sep_v0_subfigures')
    train_jsonl_path = os.path.join(pmcoa_dir, 'train.jsonl')

    save_train_path = os.path.join(save_dir, 'pmcoa_train.json')

    # read jsonl
    train_data = read_jsonl(train_jsonl_path)

    with open(save_train_path, 'w') as json_file:
        json.dump(convert(train_data, image_key='image', image_dir=image_dir, text_key='caption', random_instruction=True), json_file, indent=4)


def convert_slake(slake_dir, save_dir):
    image_dir = os.path.join(slake_dir, 'imgs')
    train_json_path = os.path.join(slake_dir, 'train_text.json')
    val_json_path = os.path.join(slake_dir, 'validation_text.json')
    test_json_path = os.path.join(slake_dir, 'test_text.json')

    save_train_path = os.path.join(save_dir, 'slake_train.json')
    save_val_path = os.path.join(save_dir, 'slake_val.json')
    save_test_path = os.path.join(save_dir, 'slake_test.json')

    # read json
    with open(train_json_path, 'r', encoding='utf-8') as f:
        train_data = json.loads(f.read())
    with open(val_json_path, 'r', encoding='utf-8') as f:
        val_data = json.loads(f.read())
    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_data = json.loads(f.read())

    with open(save_train_path, 'w') as json_file:
        json.dump(convert(train_data, image_key='img_name', image_dir=image_dir, random_instruction=True), json_file, indent=4)
    with open(save_val_path, 'w') as json_file:
        json.dump(convert(val_data, image_key='img_name', image_dir=image_dir), json_file, indent=4)
    with open(save_test_path, 'w') as json_file:
        json.dump(convert(test_data, image_key='img_name', image_dir=image_dir), json_file, indent=4)


def convert_vqarad(vqarad_dir, save_dir):
    image_dir = os.path.join(vqarad_dir, 'VQA_RAD Image Folder')
    train_json_path = os.path.join(vqarad_dir, 'train_text.json')
    test_json_path = os.path.join(vqarad_dir, 'test_text.json')

    save_train_path = os.path.join(save_dir, 'vqarad_train.json')
    save_test_path = os.path.join(save_dir, 'vqarad_test.json')

    # read json
    with open(train_json_path, 'r', encoding='utf-8') as f:
        train_data = json.loads(f.read())
    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_data = json.loads(f.read())

    with open(save_train_path, 'w') as json_file:
        json.dump(convert(train_data, image_key='image_name', image_dir=image_dir, random_instruction=True), json_file, indent=4)
    with open(save_test_path, 'w') as json_file:
        json.dump(convert(test_data, image_key='image_name', image_dir=image_dir), json_file, indent=4)


def convert_textplus(pmcvqa_text_dir, save_dir, include_pmcoa_using_path='', include_patients_using_path=''):
    pmcvqa_image_dir = os.path.join(pmcvqa_text_dir, 'images')
    rocov2_image_dir = os.path.join(pmcvqa_text_dir, 'train')
    med60k_image_dir = os.path.join(pmcvqa_text_dir, 'images_med60k', 'images')
    vision_image_dir = os.path.join(pmcvqa_text_dir, 'images_vision')

    pmcvqa_train_json_path = os.path.join(pmcvqa_text_dir, 'train_text.json')
    rocov2_train_csv_path = os.path.join(pmcvqa_text_dir, 'train_captions.csv')
    med60k_train_json_path = os.path.join(pmcvqa_text_dir, 'train_text_med60k.json')
    vision_train_json_path = os.path.join(pmcvqa_text_dir, 'PubMedVision_Alignment_VQA.json')

    save_train_path = os.path.join(save_dir, 'textplus_train.json')

    print('Reading PMC-VQA-text ...')
    with open(pmcvqa_train_json_path, 'r', encoding='utf-8') as f:
        pmcvqa_train_data = json.loads(f.read())

    print('Reading ROCOv2 ...')
    rocov2_train_data = []
    rocov2_train_data_raw = pd.read_csv(rocov2_train_csv_path)
    for id, caption in zip(rocov2_train_data_raw['ID'], rocov2_train_data_raw['Caption']):
        rocov2_train_data.append({'Figure_path': f'{id}.jpg', 'text': caption})

    print('Reading LLaVA-Med-60K-IM-text ...')
    with open(med60k_train_json_path, 'r', encoding='utf-8') as f:
        med60k_train_data = json.loads(f.read())

    print('Reading PubMedVision ...')
    print('Filter diagrams in advance.')
    vision_filenames = [f for f in sorted(os.listdir(vision_image_dir))]

    # read json
    with open(vision_train_json_path, 'r', encoding='utf-8') as f:
        vision_train_data_raw = json.loads(f.read())
    vision_dict = {}
    for elem in vision_train_data_raw:
        for im in elem['image']:
            vision_dict[os.path.basename(im)] = elem['conversations'][-1]['value']

    # filter and construct data
    vision_train_data = []
    for filename in vision_filenames:
        if filename in vision_dict:
            vision_train_data.append({'Figure_path': filename, 'text': vision_dict[filename]})

    converted_data = []
    converted_data.extend(convert(pmcvqa_train_data, image_key='Figure_path', image_dir=pmcvqa_image_dir, random_instruction=True))
    converted_data.extend(convert(rocov2_train_data, image_key='Figure_path', image_dir=rocov2_image_dir, random_instruction=True))
    converted_data.extend(convert(med60k_train_data, image_key='image', image_dir=med60k_image_dir, random_instruction=True))
    converted_data.extend(convert(vision_train_data, image_key='Figure_path', image_dir=vision_image_dir, random_instruction=True))

    print('Save json', len(converted_data), 'lines ...')
    with open(save_train_path, 'w') as json_file:
        json.dump(converted_data, json_file, indent=4)

    if include_pmcoa_using_path != '':
        print('Reading PMC-OA ...')
        pmcoa_image_dir = os.path.join(include_pmcoa_using_path, 'caption_T060_filtered_top4_sep_v0_subfigures')
        pmcoa_train_data = read_jsonl(os.path.join(include_pmcoa_using_path, 'train.jsonl'))
        converted_data.extend(convert(pmcoa_train_data, image_key='image', image_dir=pmcoa_image_dir, text_key='caption', random_instruction=True))
        save_train_path = save_train_path.replace('_train.json', '_pmcoa_train.json')

        print('Save json', len(converted_data), 'lines ...')
        with open(save_train_path, 'w') as json_file:
            json.dump(converted_data, json_file, indent=4)

    if include_patients_using_path != '':
        print('Reading PMC-Patients ...')
        patients_image_dir = os.path.join(pmcvqa_text_dir, 'images_patients', 'images')
        pmc_patients_csv_path = os.path.join(pmcvqa_text_dir, 'PMC-Patients.csv')
        id_patient_dict = {}
        for index, row in list(pd.read_csv(pmc_patients_csv_path).iterrows()):
            file_path = row['file_path']
            patient = row['patient']
            id = file_path.split('/')[-1].split('.')[0]  # PMCxxxxxxx
            if id not in id_patient_dict:
                id_patient_dict[id] = []
            id_patient_dict[id].append(patient)

        patients_train_data = []
        for f in sorted(os.listdir(patients_image_dir)):
            if not f.endswith('.jpg'):
                continue
            id = f.split('_')[0]  # PMCxxxxxxx
            for text in id_patient_dict[id]:
                patients_train_data.append({'Figure_path': f, 'text': text})

        converted_data.extend(convert(patients_train_data, image_key='Figure_path', image_dir=patients_image_dir, random_instruction=True, instruction_list=INSTRUCTION_CASE_LIST))

        save_train_path = save_train_path.replace('_train.json', '_patients_train.json')

        print('Save json', len(converted_data), 'lines ...')
        with open(save_train_path, 'w') as json_file:
            json.dump(converted_data, json_file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='textplus')
    parser.add_argument('--data_dir', default='./datasets')
    parser.add_argument('--save_dir', default='./datasets/medcap')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.dataset == 'pmcoa':
        print('Download PMC-OA in advance.')
        convert_pmcoa(os.path.join(args.data_dir, 'pmc_oa'), args.save_dir)
    elif args.dataset == 'slake':
        print('Download SLAKE and SLAKE-text in advance.')
        convert_slake(os.path.join(args.data_dir, 'SLAKE'), args.save_dir)
    elif args.dataset == 'vqarad':
        print('Download VQA-RAD and VQA-RAD-text in advance.')
        convert_vqarad(os.path.join(args.data_dir, 'VQA_RAD'), args.save_dir)
    elif args.dataset == 'textplus':
        print('Download PMC-VQA-text, ROCOv2, LLaVA-Med-60K-IM-text, PubMedVision, PMC-OA, PMC-Patients, and PMC-Patients-images in advance.')
        convert_textplus(os.path.join(args.data_dir, 'PMC-VQA-text'), args.save_dir, os.path.join(args.data_dir, 'pmc_oa'), os.path.join(args.data_dir, 'PMC-VQA-text'))
