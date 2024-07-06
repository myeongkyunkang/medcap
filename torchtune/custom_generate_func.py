import datetime
import json
import os
from copy import deepcopy

import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO

from torchtune.datasets._chat import chat_dataset

MRI_TOKENS = '①' * 32

VQA_CACHE = {}


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


def post(generated_text, all_special_tokens):
    for s_t in all_special_tokens:  # to remove <| ...
        generated_text = generated_text.split(s_t)[0]
    return generated_text.strip()


def test_metrics(recipe, cfg):
    ds = chat_dataset(
        tokenizer=recipe._tokenizer,
        source=cfg.dataset_source,
        conversation_style=cfg.dataset_conversation_style,
        chat_format=cfg.dataset_chat_format,
        max_seq_len=cfg.dataset_max_seq_len,
        train_on_input=True,
    )

    out_dict = {'image': [], 'instruction': [], 'output': [], 'generated': []}
    for sample in ds._data:
        _sample = deepcopy(sample)

        # make prompt
        output = _sample['conversations'][1]['value']
        _sample['conversations'][1]['value'] = ''  # remove answer
        tokens, _, feat = ds._prepare_sample(_sample)
        cfg.prompt = ds._tokenizer.decode(tokens)
        if cfg.prompt.endswith('<|eot_id|>'):
            cfg.prompt = cfg.prompt[:-len('<|eot_id|>')]  # remove <|eot_id|>

        if feat is not None:
            feat = feat.to(device=recipe._device, dtype=recipe._dtype).unsqueeze(0)  # set batch size 1

        generated_text = post(recipe.generate(cfg=cfg, feat=feat), recipe._tokenizer.all_special_tokens)
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

    _sample['image'] = ''
    _sample['meta'] = ''

    # make prompt
    _sample['conversations'][0]['value'] = question
    _sample['conversations'][1]['value'] = ''  # remove answer
    tokens, _, _ = ds._prepare_sample(_sample)
    cfg.prompt = ds._tokenizer.decode(tokens)
    if cfg.prompt.endswith('<|eot_id|>'):
        cfg.prompt = cfg.prompt[:-len('<|eot_id|>')]  # remove <|eot_id|>

    generated_text = post(recipe.generate(cfg=cfg), recipe._tokenizer.all_special_tokens)

    return generated_text, cfg.prompt


def vqa(question, filename, image_dir, sample, recipe, cfg, ds, instruction_describe='Describe the image in a detailed and informative manner.'):
    ori_max_new_tokens = cfg.max_new_tokens  # save max_new_tokens
    cfg.max_new_tokens = cfg.max_new_tokens // 2  # reduce max_new_tokens by half

    _sample = deepcopy(sample)

    _sample['image'] = filename
    _sample['meta'] = f'dir={image_dir}'

    # make prompt
    _sample['conversations'][0]['value'] = MRI_TOKENS + f'\n\n{instruction_describe}'
    _sample['conversations'][1]['value'] = ''  # remove answer
    tokens, _, feat = ds._prepare_sample(_sample)
    cfg.prompt = ds._tokenizer.decode(tokens)
    if cfg.prompt.endswith('<|eot_id|>'):
        cfg.prompt = cfg.prompt[:-len('<|eot_id|>')]  # remove <|eot_id|>

    if feat is not None:
        feat = feat.to(device=recipe._device, dtype=recipe._dtype).unsqueeze(0)

    global VQA_CACHE
    if f'{image_dir}/{filename}' in VQA_CACHE:
        generated_text = VQA_CACHE[f'{image_dir}/{filename}']
    else:
        generated_text = post(recipe.generate(cfg=cfg, feat=feat.clone()), recipe._tokenizer.all_special_tokens)
        VQA_CACHE[f'{image_dir}/{filename}'] = generated_text

    # make prompt
    _sample['conversations'][1]['value'] = generated_text
    _sample['conversations'].append({"from": "human", "value": question})
    _sample['conversations'].append({"from": "gpt", "value": ""})
    tokens, _, feat = ds._prepare_sample(_sample)
    cfg.prompt = ds._tokenizer.decode(tokens)
    if cfg.prompt.endswith('<|eot_id|>'):
        cfg.prompt = cfg.prompt[:-len('<|eot_id|>')]  # remove <|eot_id|>

    if feat is not None:
        feat = feat.to(device=recipe._device, dtype=recipe._dtype).unsqueeze(0)

    generated_text = post(recipe.generate(cfg=cfg, feat=feat), recipe._tokenizer.all_special_tokens)

    cfg.max_new_tokens = ori_max_new_tokens  # rollback max_new_tokens

    return generated_text, cfg.prompt


def eval_vqa(recipe, cfg, answer_dict):
    cfg.max_new_tokens = 20  # force to reduce max_new_tokens

    # read dummy dataset
    ds = chat_dataset(
        tokenizer=recipe._tokenizer,
        source='recipes/configs/dummy.json',
        conversation_style=cfg.dataset_conversation_style,
        chat_format=cfg.dataset_chat_format,
        max_seq_len=cfg.dataset_max_seq_len,
        train_on_input=True,
    )
    sample_0 = deepcopy(ds._data[0])

    correct_list = []
    out_dict = {'image': [], 'instruction': [], 'generated': []}
    for image, inst, out, gen in zip(answer_dict['image'], answer_dict['instruction'], answer_dict['output'], answer_dict['generated']):
        vqa_question = inst.split('<|eot_id|><|start_header_id|>user<|end_header_id|>')[-1].split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')[0].strip()
        question = f"Respond with only RIGHT or WRONG to indicate whether we predicted it correctly.\n\nQuestion: {vqa_question}\nOur answer: {gen}\nCorrect answer: {out}"
        generated_text, prompt = chat(question, sample_0, recipe, cfg, ds)

        print('=' * 20)
        print(image)
        print(prompt)
        print(generated_text)

        generated_text_lower = generated_text.lower()
        if 'right' in generated_text_lower and 'wrong' in generated_text_lower:
            correct_list.append(0)  # Solved via 70B
        elif 'right' in generated_text_lower:
            correct_list.append(1)
        elif 'wrong' in generated_text_lower:
            correct_list.append(0)
        else:
            correct_list.append(0)  # Solved via 70B

        out_dict['image'].append(image)
        out_dict['instruction'].append(prompt)
        out_dict['generated'].append(generated_text)

    correct_list = np.array(correct_list)
    print(f' * Acc: {np.mean(correct_list)}')

    return correct_list, out_dict


def test_vqaslake(recipe, cfg):
    json_path = os.path.join(cfg.dataset_source, 'test.json')
    image_dir = os.path.join(cfg.dataset_source, 'imgs')

    with open(json_path, "r", encoding='utf-8') as f:
        json_data = json.loads(f.read())

    # read dummy dataset
    ds = chat_dataset(
        tokenizer=recipe._tokenizer,
        source='recipes/configs/dummy.json',
        conversation_style=cfg.dataset_conversation_style,
        chat_format=cfg.dataset_chat_format,
        max_seq_len=cfg.dataset_max_seq_len,
        train_on_input=True,
    )
    sample_0 = deepcopy(ds._data[0])

    print('Test only with closed-answer types.')
    out_dict = {'image': [], 'instruction': [], 'output': [], 'generated': []}
    for elem in json_data:
        if elem['q_lang'] != 'en' or elem['answer_type'] != 'CLOSED':
            continue

        generated_text, prompt = vqa(elem['question'],
                                     filename=elem['img_name'], image_dir=image_dir,
                                     sample=sample_0, recipe=recipe, cfg=cfg, ds=ds)

        print('=' * 20)
        print(elem['img_name'])
        print(prompt)
        print(generated_text)
        print('Correct answer:', elem['answer'])

        out_dict['image'].append(elem['img_name'])
        out_dict['instruction'].append(prompt)
        out_dict['output'].append(elem['answer'])
        out_dict['generated'].append(generated_text)

    return out_dict


def test_vqarad(recipe, cfg):
    json_path = os.path.join(cfg.dataset_source, 'VQA_RAD Dataset Public.json')
    image_dir = os.path.join(cfg.dataset_source, 'VQA_RAD Image Folder')

    with open(json_path, "r", encoding='utf-8') as f:
        json_data = json.loads(f.read())

    # read dummy dataset
    ds = chat_dataset(
        tokenizer=recipe._tokenizer,
        source='recipes/configs/dummy.json',
        conversation_style=cfg.dataset_conversation_style,
        chat_format=cfg.dataset_chat_format,
        max_seq_len=cfg.dataset_max_seq_len,
        train_on_input=True,
    )
    sample_0 = deepcopy(ds._data[0])

    print('Test only with closed-answer types.')
    out_dict = {'image': [], 'instruction': [], 'output': [], 'generated': []}
    for elem in json_data:
        if (elem['phrase_type'] not in ['test_freeform', 'test_para']) or (elem['answer_type'] != 'CLOSED'):
            continue

        generated_text, prompt = vqa(elem['question'],
                                     filename=elem['image_name'], image_dir=image_dir,
                                     sample=sample_0, recipe=recipe, cfg=cfg, ds=ds)

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
