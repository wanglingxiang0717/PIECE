import re
from rouge import Rouge
from fuzzywuzzy import fuzz
from datasets import load_metric
from nltk.translate.bleu_score import sentence_bleu

import json
# import jsonl
from tqdm import tqdm
import os
import pandas as pd

def read_json_or_jsonl(file_path):
    data = []
    txt = file_path.split('/')[-1].split('.')[-1]
    if txt == 'json':
        with open(file_path, 'r') as f_r:
            data = json.load(f_r)
    elif txt == 'jsonl':
        with open(file_path, 'r') as f_r:
            for line in f_r:
                data.append(json.loads(line))
    else:
        data = None
    return data

def write_json_or_jsonl(file_path, data, encoding='utf-8'):
    txt = file_path.split('/')[-1].split('.')[-1]
    if txt == 'json':
        with open(file_path, 'w', encoding=encoding) as f_w:
            json.dump(data, f_w, ensure_ascii=False, indent=2)
    elif txt == 'jsonl':
        with open(file_path, 'w', encoding=encoding) as f_w:
            for line in data:
                json_line = json.dumps(line)
                f_w.write(json_line + '\n')
    else:
        print("file_path not exisits")

def tokenize(text):
    tokens = re.split(r'\s|\.', text)
    tokens = [t for t in tokens if len(t) > 0]
    return tokens


def bleu_score(reference, hypothesis, gram):
    reference_tokens = tokenize(reference)
    hypothesis_tokens = tokenize(hypothesis)

    if gram == 1:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1., ))  # BELU-1
    elif gram == 2:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 2., 1. / 2.))  # BELU-2
    elif gram == 3:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 3., 1. / 3., 1. / 3.))  # BELU-3
    elif gram == 4:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 4., 1. / 4., 1. / 4., 1. / 4.))  # BELU-4

    return bleu


def caculate_bleu(results, data, gram):
    bleus = []
    for output_id in range(len(results)):
        prediction = results[output_id]
        target = data[output_id] 
        if isinstance(target, list):
            bleu = 0
            for tar in target:
                bt = bleu_score(tar, prediction, gram)
                if bt > bleu:
                    bleu = bt
        else:
            if prediction == "" or target == "":
                continue
            bleu = bleu_score(target, prediction, gram)
        bleus.append(bleu)
    avg_bleu = sum(bleus) / len(results)
    return avg_bleu

def score_rouge(str1, str2):
    rouge = Rouge(metrics=["rouge-l"])
    scores = rouge.get_scores(str1, str2, avg=True)
    rouge_l = scores['rouge-l']['f']
    return rouge_l


def caculate_rouge(results, data):
    rouges = []
    for output_id in range(len(results)):
        prediction = results[output_id]
        target = data[output_id] 
        if isinstance(target, list):
            rouge = 0
            for tar in target:
                rt = score_rouge(tar, prediction)
                if rt > rouge:
                    rouge = rt
        else:
            if prediction == "" or target == "":
                continue
            rouge = score_rouge(target, prediction)
        rouges.append(rouge)
    avg_rouge = sum(rouges) / len(results)
    return avg_rouge

def eval(predicted_sequences, ground_truths):
    bleu_1 = caculate_bleu(predicted_sequences, ground_truths, 1)
    bleu_4 = caculate_bleu(predicted_sequences, ground_truths, 4)
    rouge = caculate_rouge(predicted_sequences, ground_truths)
    # print("blue_1:", bleu_1, "bleu_4:", bleu_4, "rouge:", rouge)
    evaluation_result = {"bleu-1": bleu_1, "bleu-4": bleu_4, "rouge-L": rouge}
    print(evaluation_result)
    return evaluation_result

def main():
    file_path = "/data1/TAP/model_mlm_save_new/base/action/2/flickr30k_multibatch_result.json"
    data = read_json_or_jsonl(file_path)

    predic = []
    ground = []
    for item in data:
        predic.append(item['answer'])
        ground.append(item['label'])
    result = eval(predic, ground)
    print(result)

if __name__ == "__main__":
    main()