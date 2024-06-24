import json
import os
import re
from copy import deepcopy

import nltk
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score, precision_score, recall_score

from torchtune.datasets._chat import chat_dataset

nltk.download('punkt')

MRI_TOKENS = '①' * 32


def calculate_f1(reference_tokens, candidate_tokens):
    ref_set = set(reference_tokens)
    cand_set = set(candidate_tokens)

    all_tokens = list(ref_set.union(cand_set))

    y_true = [1 if token in ref_set else 0 for token in all_tokens]
    y_pred = [1 if token in cand_set else 0 for token in all_tokens]

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1


def calculate_text_metrics(reference, candidate):
    reference = re.sub(r'\s+', ' ', reference).strip()
    candidate = re.sub(r'\s+', ' ', candidate).strip()

    reference_tokens = nltk.word_tokenize(reference)
    candidate_tokens = nltk.word_tokenize(candidate)

    reference_text = " ".join(reference_tokens)
    candidate_text = " ".join(candidate_tokens)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_text, candidate_text)
    rouge1_score = scores['rouge1'].fmeasure
    rouge2_score = scores['rouge2'].fmeasure
    rougeL_score = scores['rougeL'].fmeasure

    smoothing = SmoothingFunction().method1

    bleu1_score = sentence_bleu([reference_tokens], candidate_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2_score = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3_score = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu4_score = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

    gleu_score = sentence_gleu([reference_tokens], candidate_tokens)

    precision, recall, f1 = calculate_f1(reference_tokens, candidate_tokens)

    return {
        'ROUGE-1': rouge1_score,
        'ROUGE-2': rouge2_score,
        'ROUGE-L': rougeL_score,
        'BLEU-1': bleu1_score,
        'BLEU-2': bleu2_score,
        'BLEU-3': bleu3_score,
        'BLEU-4': bleu4_score,
        'GLEU': gleu_score,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }


def test_metrics(recipe, cfg):
    ds = chat_dataset(
        tokenizer=recipe._tokenizer,
        source=cfg.dataset_source,
        conversation_style=cfg.dataset_conversation_style,
        chat_format=cfg.dataset_chat_format,
        max_seq_len=cfg.dataset_max_seq_len,
        train_on_input=True,
    )

    out_dict = {'instruction': [], 'output': [], 'image': [], 'generated': []}
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

        def post(generated_text):
            for s_t in recipe._tokenizer.all_special_tokens:  # to remove <| ...
                generated_text = generated_text.split(s_t)[0]
            return generated_text.strip()

        generated_text = post(recipe.generate(cfg=cfg, feat=feat))
        print('=' * 20)
        print(generated_text)

        out_dict['instruction'].append(cfg.prompt)
        out_dict['output'].append(output)
        out_dict['image'].append(_sample['image'])
        out_dict['generated'].append(generated_text)

    # write generated
    df = pd.DataFrame(out_dict)
    df.to_csv(os.path.join(cfg.output_dir, f'test_out_{cfg.checkpointer.checkpoint_files[0].split(".")[0]}.csv'), index=False, encoding='utf-8-sig')

    # calc metrics
    text_metrics_list = {}
    for output, generated in zip(out_dict['output'], out_dict['generated']):
        text_metrics = calculate_text_metrics(output.lower(), generated.strip().lower())
        for k, v in text_metrics.items():
            if k not in text_metrics_list:
                text_metrics_list[k] = []
            text_metrics_list[k].append(v)

    # write metrics
    with open(os.path.join(cfg.output_dir, f'test_{cfg.checkpointer.checkpoint_files[0].split(".")[0]}.csv'), 'wt') as f:
        for k, v in text_metrics_list.items():
            print(f"{k}: {np.mean(v)}")
            f.write(f"{k},{np.mean(v)}\n")


def test_vqarad(recipe, cfg):
    json_path = os.path.join(cfg.dataset_source, 'VQA_RAD Dataset Public.json')
    image_dir = os.path.join(cfg.dataset_source, 'VQA_RAD Image Folder')

    with open(json_path, "r") as f:
        json_data = json.loads(f.read())

    # read dataset
    ds = chat_dataset(
        tokenizer=recipe._tokenizer,
        source='recipes/configs/dummy.json',
        conversation_style=cfg.dataset_conversation_style,
        chat_format=cfg.dataset_chat_format,
        max_seq_len=cfg.dataset_max_seq_len,
        train_on_input=True,
    )
    sample_0 = deepcopy(ds._data[0])

    out_dict = {'instruction': [], 'output': [], 'image': [], 'generated': []}
    for elem in json_data:
        _sample = deepcopy(sample_0)

        image_name = elem['image_name']
        question = elem['question']
        answer = elem['answer']

        _sample['image'] = image_name
        _sample['meta'] = f'dir={image_dir}'

        # make prompt
        _sample['conversations'][0]['value'] = MRI_TOKENS + '\n\nDescribe the image in a detailed and informative manner.'
        _sample['conversations'][1]['value'] = ''  # remove answer
        ds.max_seq_len = cfg.dataset_max_seq_len  # reset max_seq_len
        tokens, _, feat = ds._prepare_sample(_sample)
        cfg.prompt = ds._tokenizer.decode(tokens)
        if cfg.prompt.endswith('<|eot_id|>'):
            cfg.prompt = cfg.prompt[:-len('<|eot_id|>')]  # remove <|eot_id|>

        if feat is not None:
            feat = feat.to(device=recipe._device, dtype=recipe._dtype).unsqueeze(0)

        def post(generated_text):
            for s_t in recipe._tokenizer.all_special_tokens:  # to remove <| ...
                generated_text = generated_text.split(s_t)[0]
            return generated_text.strip()

        generated_text = post(recipe.generate(cfg=cfg, feat=feat.clone()))

        # make prompt
        _sample['conversations'][1]['value'] = generated_text
        _sample['conversations'].append({"from": "human", "value": question})
        _sample['conversations'].append({"from": "gpt", "value": ""})
        ds.max_seq_len = cfg.dataset_max_seq_len * 2  # increase max_seq_len
        tokens, _, feat = ds._prepare_sample(_sample)
        cfg.prompt = ds._tokenizer.decode(tokens)
        if cfg.prompt.endswith('<|eot_id|>'):
            cfg.prompt = cfg.prompt[:-len('<|eot_id|>')]  # remove <|eot_id|>

        if feat is not None:
            feat = feat.to(device=recipe._device, dtype=recipe._dtype).unsqueeze(0)

        generated_text = post(recipe.generate(cfg=cfg, feat=feat))
        print('=' * 20)
        print(cfg.prompt)
        print(generated_text)
        print('Correct answer:', answer)

        out_dict['instruction'].append(cfg.prompt)
        out_dict['image'].append(_sample['image'])
        out_dict['generated'].append(generated_text)
        out_dict['output'].append(answer)

    # write generated
    df = pd.DataFrame(out_dict)
    df.to_csv(os.path.join(cfg.output_dir, f'test_vqarad_out_{cfg.checkpointer.checkpoint_files[0].split(".")[0]}.csv'), index=False, encoding='utf-8-sig')
