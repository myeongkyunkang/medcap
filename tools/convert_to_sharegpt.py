import argparse
import json
import os
import random

import pandas as pd

random.seed(0)

MRI_TOKENS = '①' * 50
INSTRUCTION_LIST = [
    "Describe the image in a detailed and informative manner.", "Summarize the visual content of the image clearly.", "Provide a rich, descriptive narrative of the image.", "Offer a brief but comprehensive description of the image.", "Analyze the image thoroughly and in detail.", "Narrate the contents of the image with precision.", "Give a detailed account of the given image.", "Describe the following image in detail.", "Examine the image closely and share its details.",
    "Give a short, clear explanation of the image.", "Break down the elements of the image in detail.", "Relay a brief, clear account of the image shown.", "Share a concise interpretation of the image provided.", "Write a short but informative summary of the image.", "Walk through the important details of the image.", "Explain the various aspects of the image.", "Describe the image concisely.", "Offer a succinct explanation of the image.",
    "Characterize the image with a well-detailed description.", "Write an exhaustive depiction of the given image.", "Clarify the contents of the displayed image in great detail.", "Render a clear and concise summary of the image.", "Give an elaborate explanation of the image you see.", "Offer a thorough analysis of the image.", "Share a comprehensive rundown of the presented image.", "Create a compact narrative representing the image.",
    "Present a concise description of the image’s key features.", "Can you provide a radiology report for this medical image?", "Describe the medical image you see.", "What is depicted in this picture?", "Please report this medical scan.", "What is the medical significance of this image?", "What can you infer from this picture?", "Can you provide a quick summary of this image?", "Describe this medical scan.", "Please write a radiology report for this image.",
    "Can you summarize the images presented?", "Please generate a radiology report for this scan.", "Describe the regions of interest in this scan.", "Please provide a caption for this medical image.", "Can you provide a brief summary of this radiograph?", "Describe the structures involved in this medical image.", "What are the findings presented in this medical scan?", "Please write a radiology report for this scan.", "Can you provide a description of this medical scan?",
    "Please caption this medical scan.", "Can you provide a report summary for this medical scan?", "Please describe this picture.", "Can you describe the image for me?", "What details stand out in this image?", "Could you provide a detailed description of what is shown in the picture?", "What is the main focus of this photograph?", "Describe the composition and the subjects in this picture.", "Explain the visual content of the image.",
    "Analyze the image in a comprehensive and detailed manner.", "Write a detailed description of the given image.", "What is this photo about?", "What is depicted in the image?",
]
INSTRUCTION_CASE_LIST = ["Can you provide a case report?", "Please write a case report.", "Please generate a case report.", "Can you provide a case summary?", ]


def convert(data, image_key, image_dir, text_key='text', instruction_type='no', instruction_list=INSTRUCTION_LIST):
    out = []
    for elem in data:
        if instruction_type == 'no':
            value = MRI_TOKENS
        elif instruction_type == 'random':
            value = MRI_TOKENS + '\n\n' + random.choice(instruction_list)
        elif instruction_type == 'fixed':
            value = MRI_TOKENS + '\n\n' + random.choice(instruction_list[0])
        else:
            raise ValueError('Invalid instruction_type:', instruction_type)
        out.append({
            'conversations': [
                {"from": "human", "value": value},
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
        json.dump(convert(train_data, image_key='image', image_dir=image_dir, text_key='caption'), json_file, indent=4)


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
        json.dump(convert(train_data, image_key='image_name', image_dir=image_dir), json_file, indent=4)
    with open(save_test_path, 'w') as json_file:
        json.dump(convert(test_data, image_key='image_name', image_dir=image_dir), json_file, indent=4)


def convert_textplus(pmcvqa_text_dir, save_dir, include_pmcoa_using_path='', include_patients_using_path='', save_filename='textplus_train.json'):
    pmcvqa_image_dir = os.path.join(pmcvqa_text_dir, 'images')
    rocov2_image_dir = os.path.join(pmcvqa_text_dir, 'train')
    med60k_image_dir = os.path.join(pmcvqa_text_dir, 'images_med60k', 'images')
    vision_image_dir = os.path.join(pmcvqa_text_dir, 'images_vision')

    pmcvqa_train_json_path = os.path.join(pmcvqa_text_dir, 'train_text.json')
    rocov2_train_csv_path = os.path.join(pmcvqa_text_dir, 'train_captions.csv')
    med60k_train_json_path = os.path.join(pmcvqa_text_dir, 'train_text_med60k.json')
    vision_train_json_path = os.path.join(pmcvqa_text_dir, 'PubMedVision_Alignment_VQA.json')

    save_train_path = os.path.join(save_dir, save_filename)

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
    converted_data.extend(convert(pmcvqa_train_data, image_key='Figure_path', image_dir=pmcvqa_image_dir))
    converted_data.extend(convert(rocov2_train_data, image_key='Figure_path', image_dir=rocov2_image_dir))
    converted_data.extend(convert(med60k_train_data, image_key='image', image_dir=med60k_image_dir))
    converted_data.extend(convert(vision_train_data, image_key='Figure_path', image_dir=vision_image_dir))

    print('Save json', len(converted_data), 'lines ...')
    with open(save_train_path, 'w') as json_file:
        json.dump(converted_data, json_file, indent=4)

    medicat_image_dir = os.path.join(pmcvqa_text_dir, 'release', 'figures')
    medicat_train_jsonl_path = os.path.join(pmcvqa_text_dir, 'release', 's2_full_figures_oa_nonroco_combined_medical_top4_public.jsonl')
    if os.path.isfile(medicat_train_jsonl_path):
        print('Reading MediCaT ...')
        medicat_train_data_raw = read_jsonl(medicat_train_jsonl_path)

        # filter and construct data
        medicat_train_data = []
        for elem in medicat_train_data_raw:
            filename = f"{elem['pdf_hash']}_{elem['fig_uri']}"
            if filename == 'ffd83b6453f94f2a1ddb346e324f5bdbf228c1f3_4-Figure3-1.png':
                continue
            text = elem['s2_caption']
            if elem.get('s2orc_references', None) is not None:
                text = text + ' ' + ' '.join(elem['s2orc_references'])
            medicat_train_data.append({'Figure_path': filename, 'text': text})

        converted_data.extend(convert(medicat_train_data, image_key='Figure_path', image_dir=medicat_image_dir))
    else:
        print('Skip MediCaT')

    if include_pmcoa_using_path != '':
        print('Reading PMC-OA ...')
        pmcoa_image_dir = os.path.join(include_pmcoa_using_path, 'caption_T060_filtered_top4_sep_v0_subfigures')
        pmcoa_train_data = read_jsonl(os.path.join(include_pmcoa_using_path, 'train.jsonl'))
        converted_data.extend(convert(pmcoa_train_data, image_key='image', image_dir=pmcoa_image_dir, text_key='caption'))
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

        converted_data.extend(convert(patients_train_data, image_key='Figure_path', image_dir=patients_image_dir))

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
    elif args.dataset == 'vqarad':
        print('Download VQA-RAD and VQA-RAD-text in advance.')
        convert_vqarad(os.path.join(args.data_dir, 'VQA_RAD'), args.save_dir)
    elif args.dataset == 'textplus':
        print('Download PMC-VQA-text, ROCOv2, LLaVA-Med-60K-IM-text, PubMedVision, MediCaT, PMC-OA, PMC-Patients, and PMC-Patients-images in advance.')
        convert_textplus(os.path.join(args.data_dir, 'PMC-VQA-text'), args.save_dir, os.path.join(args.data_dir, 'pmc_oa'), os.path.join(args.data_dir, 'PMC-VQA-text'))
