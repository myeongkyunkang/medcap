import re

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score, precision_score, recall_score

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
