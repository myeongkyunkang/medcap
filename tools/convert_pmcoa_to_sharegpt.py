import json
import os
import random

MRI_TOKENS = '①' * 32

prompt_list = [
    "Describe the image in a detailed and informative manner.",
    "Summarize the visual content of the image clearly.",
    "Provide a rich, descriptive narrative of the image.",
    "Offer a brief but comprehensive description of the image.",
    "Analyze the image thoroughly and in detail.",
    "Narrate the contents of the image with precision.",
    "Give a detailed account of the given image.",
    "Describe the following image in detail.",
    "Examine the image closely and share its details.",
    "Give a short, clear explanation of the image.",
    "Break down the elements of the image in detail.",
    "Relay a brief, clear account of the image shown.",
    "Share a concise interpretation of the image provided.",
    "Write a short but informative summary of the image.",
    "Walk through the important details of the image.",
    "Explain the various aspects of the image.",
    "Describe the image concisely.",
    "Offer a succinct explanation of the image.",
    "Characterize the image with a well-detailed description.",
    "Write an exhaustive depiction of the given image.",
    "Clarify the contents of the displayed image in great detail.",
    "Render a clear and concise summary of the image.",
    "Give an elaborate explanation of the image you see.",
    "Offer a thorough analysis of the image.",
    "Share a comprehensive rundown of the presented image.",
    "Create a compact narrative representing the image.",
    "Present a concise description of the image’s key features.",

    "Can you provide a radiology report for this medical image?",
    "Describe the medical image you see.",
    "What is depicted in this picture?",
    "Please report this medical scan.",
    "What is the medical significance of this image?",
    "What can you infer from this picture?",
    "Can you provide a quick summary of this image?",
    "Describe this medical scan.",
    "Please write a radiology report for this image.",
    "Can you summarize the images presented?",
    "Please generate a radiology report for this scan.",
    "Describe the regions of interest in this scan.",
    "Please provide a caption for this medical image.",
    "Can you provide a brief summary of this radiograph?",
    "Describe the structures involved in this medical image.",
    "What are the findings presented in this medical scan?",
    "Please write a radiology report for this scan.",
    "Can you provide a description of this medical scan?",
    "Please caption this medical scan.",
    "Can you provide a report summary for this medical scan?",
]


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
    save_val_path = os.path.join(save_dir, 'pmcoa_valid.json')
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
                    {"from": "human", "value": MRI_TOKENS + '\n\n' + random.choice(prompt_list)},
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
