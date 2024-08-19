import datetime
import json
import os
import random
from copy import deepcopy

import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO
from tqdm import tqdm

from torchtune.datasets._chat import chat_dataset

MRI_TOKENS = '①' * 50

VQA_CACHE = {}


def sample_to_prompt(_sample, ds, calc_next_num_token=False):
    _sample_prepared = ds._prepare_sample(_sample)
    image = _sample_prepared.get("image", None)
    prompt = ds._tokenizer.decode(_sample_prepared['tokens'])
    if prompt.endswith('<|eot_id|>'):
        prompt = prompt[:-len('<|eot_id|>')]  # remove <|eot_id|>        

    if calc_next_num_token:
        __sample = deepcopy(_sample)
        __sample['image'], __sample['meta'] = '', ''  # for faster _prepare_sample
        __sample['conversations'].append({"from": "human", "value": ""})
        __sample['conversations'].append({"from": "gpt", "value": ""})
        _prompt, _ = sample_to_prompt(__sample, ds, calc_next_num_token=False)
        return prompt, image, len(ds._tokenizer.encode(_prompt))

    return prompt, image


class _COCOEvalCap(COCOEvalCap):
    def evaluate(self):
        imgIds, gts, res = self.params['image_id'], {}, {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]
        tokenizer = PTBTokenizer()
        gts, res = tokenizer.tokenize(gts), tokenizer.tokenize(res)
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE")  # UPDATED
        ]
        for scorer, method in scorers:
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
        self.setEvalImgs()


def test_metrics(recipe, cfg):
    ds = chat_dataset(
        tokenizer=recipe._tokenizer,
        source=cfg.dataset_source,
        conversation_style=cfg.dataset_conversation_style,
        max_seq_len=cfg.dataset_max_seq_len,
        train_on_input=True,
    )

    out_dict = {'image': [], 'instruction': [], 'output': [], 'generated': []}
    for sample in ds._data:
        _sample = deepcopy(sample)

        # make prompt
        output = _sample['conversations'][1]['value']
        _sample['conversations'][1]['value'] = ''  # remove answer
        cfg.prompt, image = sample_to_prompt(_sample, ds)

        image = image.to(device=recipe._device, dtype=recipe._dtype).unsqueeze(0) if image is not None else None

        generated_text = recipe.generate(cfg=cfg, image=image)
        print('=' * 20)
        print(_sample['image'])
        print(generated_text)

        out_dict['image'].append(_sample['image'])
        out_dict['instruction'].append(cfg.prompt)
        out_dict['output'].append(output)
        out_dict['generated'].append(generated_text)

    # calc metrics
    ref_path, gen_path = os.path.join(cfg.output_dir, f'.tmp ref {datetime.datetime.now()}.json'), os.path.join(cfg.output_dir, f'.tmp gen {datetime.datetime.now()}.json')
    ref_data, gen_data = {'images': [], 'annotations': []}, []
    for i, (out, gen) in enumerate(zip(out_dict['output'], out_dict['generated'])):
        ref_data['images'].append({'id': f'{i}'})
        ref_data['annotations'].append({'image_id': f'{i}', 'id': f'{i}', 'caption': out})
        gen_data.append({'image_id': f'{i}', 'caption': gen})
    with open(ref_path, 'w') as json_file:
        json.dump(ref_data, json_file)
    with open(gen_path, 'w') as json_file:
        json.dump(gen_data, json_file)
    coco = COCO(ref_path)
    coco_result = coco.loadRes(gen_path)
    coco_eval = _COCOEvalCap(coco, coco_result)
    coco_eval.evaluate()
    text_metrics = coco_eval.eval.items()

    # print text_metrics
    for metric, score in text_metrics:
        print(f'{metric}: {score:.6f}')

    return text_metrics, out_dict


def chat(question, sample, recipe, cfg, ds):
    _sample = deepcopy(sample)

    # make prompt
    _sample['image'], _sample['meta'] = '', ''
    _sample['conversations'][0]['value'] = question
    _sample['conversations'][1]['value'] = ''  # remove answer
    cfg.prompt, _ = sample_to_prompt(_sample, ds)

    generated_text = recipe.generate(cfg=cfg)

    return generated_text, cfg.prompt


def vqa(question, filename, image_dir, sample, recipe, cfg, ds, vqa_type='no_cond_desc'):
    if vqa_type not in ['no_cond_desc', 'no_cond_desc_no_image', 'cond_desc', 'cond_desc_no_image', 'omit']:
        raise ValueError('Invalid vqa_type:', vqa_type)

    ori_max_new_tokens = cfg.max_new_tokens  # save max_new_tokens

    _sample = deepcopy(sample)

    # load image
    _sample['image'], _sample['meta'] = filename, f'dir={image_dir}'
    _, image = sample_to_prompt(_sample, ds)
    image = image.to(device=recipe._device, dtype=recipe._dtype).unsqueeze(0) if image is not None else None

    # make prompt
    _sample['image'], _sample['meta'] = '', ''  # for faster _prepare_sample
    _sample['conversations'][0]['value'] = MRI_TOKENS
    _sample['conversations'][1]['value'] = ''  # remove answer

    if vqa_type in ['no_cond_desc', 'no_cond_desc_no_image']:
        global VQA_CACHE
        if f'{image_dir}/{filename}' in VQA_CACHE:
            generated_text = VQA_CACHE[f'{image_dir}/{filename}']
        else:
            cfg.prompt, _, next_num_token = sample_to_prompt(_sample, ds, calc_next_num_token=True)
            cfg.max_new_tokens = min(ds.max_seq_len - (next_num_token + len(ds._tokenizer.encode(question)) + 5), ori_max_new_tokens)  # reduce max_new_tokens (5 for spares)
            generated_text = recipe.generate(cfg=cfg, image=image)
            VQA_CACHE[f'{image_dir}/{filename}'] = generated_text
    elif vqa_type in ['cond_desc', 'cond_desc_no_image']:
        __sample = deepcopy(_sample)
        __sample['conversations'][1]['value'] = '...'
        __sample['conversations'].append({"from": "human", "value": f'Describe the image to answer the question: {question}'})  # Though the number of tokens increases, MRI_TOKENS will be removed.
        __sample['conversations'].append({"from": "gpt", "value": ""})
        cfg.prompt, _, next_num_token = sample_to_prompt(__sample, ds, calc_next_num_token=True)
        cfg.max_new_tokens = min(ds.max_seq_len - (next_num_token + len(ds._tokenizer.encode(question)) + 5), ori_max_new_tokens)  # reduce max_new_tokens (5 for spares)
        generated_text = recipe.generate(cfg=cfg, image=image)
    elif vqa_type == 'omit':
        generated_text = '...'

    if vqa_type in ['no_cond_desc_no_image', 'cond_desc_no_image']:
        _sample['conversations'][0]['value'] = 'Describe the image.'  # replace MRI_TOKENS
        image = None

    # make prompt
    _sample['conversations'][1]['value'] = generated_text
    _sample['conversations'].append({"from": "human", "value": question})
    _sample['conversations'].append({"from": "gpt", "value": ""})
    cfg.prompt, _ = sample_to_prompt(_sample, ds)

    cfg.max_new_tokens = min(50, ori_max_new_tokens)  # maybe enough for an answer

    generated_text = recipe.generate(cfg=cfg, image=image)

    cfg.max_new_tokens = ori_max_new_tokens  # rollback max_new_tokens

    return generated_text, cfg.prompt


def test_vqarad(recipe, cfg):
    json_path = os.path.join(cfg.dataset_source, 'VQA_RAD', 'VQA_RAD Dataset Public.json')
    image_dir = os.path.join(cfg.dataset_source, 'VQA_RAD', 'VQA_RAD Image Folder')

    with open(json_path, "r", encoding='utf-8') as f:
        json_data = json.loads(f.read())

    # read dummy dataset
    ds = chat_dataset(
        tokenizer=recipe._tokenizer,
        source='recipes/configs/dummy.json',
        conversation_style=cfg.dataset_conversation_style,
        max_seq_len=cfg.dataset_max_seq_len,
        train_on_input=True,
    )
    sample_0 = deepcopy(ds._data[0])

    print('Test only with closed-answer types.')
    out_dict = {'image': [], 'instruction': [], 'output': [], 'generated': []}
    for elem in tqdm(json_data):
        if (elem['phrase_type'] not in ['test_freeform', 'test_para']) or (elem['answer_type'] != 'CLOSED'):
            continue

        generated_text, prompt = vqa(elem['question'],
                                     filename=elem['image_name'], image_dir=image_dir,
                                     sample=sample_0, recipe=recipe, cfg=cfg, ds=ds,
                                     vqa_type=cfg.vqa_type)

        if cfg.get('debug', False):
            print('=' * 20)
            print(elem['image_name'])
            print(prompt)
            print(generated_text)
            print('Correct answer:', elem['answer'])

        out_dict['image'].append(elem['image_name'])
        out_dict['instruction'].append(prompt)
        out_dict['output'].append(elem['answer'])
        out_dict['generated'].append(generated_text)

    return out_dict


def test_omnimedvqa(recipe, cfg):
    json_dir = os.path.join(cfg.dataset_source, 'OmniMedVQA', 'QA_information', 'Open-access')
    image_dir = os.path.join(cfg.dataset_source, 'OmniMedVQA')

    json_data = []
    for filename in os.listdir(json_dir):
        if not filename.endswith('.json'):
            continue
        with open(os.path.join(json_dir, filename), "r", encoding='utf-8') as f:
            json_data.extend(json.loads(f.read()))

    # read dummy dataset
    ds = chat_dataset(
        tokenizer=recipe._tokenizer,
        source='recipes/configs/dummy.json',
        conversation_style=cfg.dataset_conversation_style,
        max_seq_len=cfg.dataset_max_seq_len,
        train_on_input=True,
    )
    sample_0 = deepcopy(ds._data[0])

    modalities = ['CT(Computed Tomography)', 'X-Ray', 'Dermoscopy', 'Fundus Photography', 'OCT (Optical Coherence Tomography', 'ultrasound', 'MR (Mag-netic Resonance Imaging)', 'Microscopy Images']
    json_data = [d for d in json_data if d.get('modality_type', '') in modalities]
    random.Random(0).shuffle(json_data)

    test_chunk_idx, test_chunk = cfg.get('test_chunk_idx', 0), cfg.get('test_chunk', 50)
    if (test_chunk_idx == -1) or (test_chunk == -1):
        chunk_size = max(len(json_data) // test_chunk, 50)
        start_idx, end_index = chunk_size * test_chunk_idx, chunk_size * (test_chunk_idx + 1)
        if test_chunk_idx == (test_chunk - 1):
            end_index = len(json_data)
        json_data = json_data[start_idx:end_index]

    print('Test only with modalities:', modalities, 'Num testing samples:', len(json_data))

    out_dict = {'image': [], 'instruction': [], 'output': [], 'generated': [], 'option_A': [], 'option_B': [], 'option_C': [], 'option_D': [], 'modality_type': [], 'question_id': []}
    for elem in tqdm(json_data):
        generated_text, prompt = vqa(elem['question'],
                                     filename=elem['image_path'], image_dir=image_dir,
                                     sample=sample_0, recipe=recipe, cfg=cfg, ds=ds,
                                     vqa_type=cfg.vqa_type)

        if cfg.get('debug', False):
            print('=' * 20)
            print(elem['image_path'])
            print(prompt)
            print(generated_text)
            print('Correct answer:', elem['gt_answer'])

        out_dict['image'].append(elem['image_path'])
        out_dict['instruction'].append(prompt)
        out_dict['output'].append(elem['gt_answer'])
        out_dict['generated'].append(generated_text)

        out_dict['option_A'].append(elem.get('option_A', ''))
        out_dict['option_B'].append(elem.get('option_B', ''))
        out_dict['option_C'].append(elem.get('option_C', ''))
        out_dict['option_D'].append(elem.get('option_D', ''))
        out_dict['modality_type'].append(elem.get('modality_type', ''))
        out_dict['question_id'].append(elem.get('question_id', ''))

    return out_dict


def eval_vqa(recipe, cfg, answer_dict):
    ori_max_new_tokens = cfg.max_new_tokens  # save max_new_tokens

    cfg.max_new_tokens = 20  # maybe enough for an answer

    # read dummy dataset
    ds = chat_dataset(
        tokenizer=recipe._tokenizer,
        source='recipes/configs/dummy.json',
        conversation_style=cfg.dataset_conversation_style,
        max_seq_len=cfg.dataset_max_seq_len,
        train_on_input=True,
    )
    sample_0 = deepcopy(ds._data[0])

    predict_list, predict_generated_list, correct_list = [], [], []
    for i in tqdm(list(range(len(answer_dict['image'])))):
        image, inst, out, gen = answer_dict['image'][i], answer_dict['instruction'][i], answer_dict['output'][i], answer_dict['generated'][i]
        A, B, C, D = answer_dict['option_A'][i], answer_dict['option_B'][i], answer_dict['option_C'][i], answer_dict['option_D'][i]
        gt = answer_dict['output'][i]

        vqa_question = inst.split('<|eot_id|><|start_header_id|>user<|end_header_id|>')[-1].split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')[0].strip()
        if C == '' and D == '':
            question = f"Question: {vqa_question}\nOur answer: {gen}\nA. {A}\nB. {B}\n\n"
        else:
            question = f"Question: {vqa_question}\nOur answer: {gen}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\n"
        question += 'Based on the question and our answer, respond with only the letter of the correct option from the given choices.'
        generated_text, prompt = chat(question, sample_0, recipe, cfg, ds)

        try:
            predict = generated_text.strip()[0].upper()
            correct = ['A', 'B', 'C', 'D'].index(predict) == [A, B, C, D].index(gt)
        except:
            print('Error predict:', generated_text)
            correct = 0

        if cfg.get('debug', False):
            print('=' * 20)
            print(image)
            print(prompt)
            print(generated_text)
            print(['Wrong', 'Correct'][correct])

        predict_list.append(predict)
        predict_generated_list.append(generated_text)
        correct_list.append(1 if correct else 0)

    answer_dict['predict_generated'] = predict_generated_list
    answer_dict['predict'] = predict_list
    answer_dict['correct'] = correct_list

    modality_type_list = sorted(list(set(answer_dict['modality_type'])))
    acc_per_modality_type = []
    for modality_type in modality_type_list:
        correct_list_per_modality = [c for c, m in zip(answer_dict['correct'], answer_dict['modality_type']) if m == modality_type]
        acc_per_modality_type.append(np.mean(correct_list_per_modality))
    acc = np.mean(acc_per_modality_type)

    for m, a in zip(modality_type_list, acc_per_modality_type):
        print(f'{m}: {round(a, 4)}')

    print(f' * Acc: {acc}')

    cfg.max_new_tokens = ori_max_new_tokens  # rollback max_new_tokens

    return acc, answer_dict
