import json
import jsonlines
from metrics import caculate_bleu, caculate_rouge, caculate_accuracy

#add by wlx
def find_first_occurrence(s):
    target = {'A', 'B', 'C', 'D'}
    return next((char for char in s if char in target), None)

# resolving answer and reasoning
# def resolve(dataset: list):
#     answers = []
#     reasonings = []
#     for datium in dataset:
#         answers.append(datium[0]) # the first char is the answer. e.g. A, B,...
#         reasonings.append(datium[2:]) # A/nBecause...
#     outputs = {"answers": answers, "reasonings": reasonings}
#     return outputs

def resolve(dataset: list):
    answers = []
    reasonings = []
    for datium in dataset:
        answers.append(find_first_occurrence(datium)) # the first char is the answer. e.g. A, B,...
        reasonings.append(datium) # A/nBecause...
    outputs = {"answers": answers, "reasonings": reasonings}
    return outputs

def eval(predicted_sequences, ground_truths):
    outputs = resolve(predicted_sequences)
    gts = resolve(ground_truths)

    bleu_1 = caculate_bleu(outputs["reasonings"], gts["reasonings"], 1)
    bleu_4 = caculate_bleu(outputs["reasonings"], gts["reasonings"], 4)
    rouge = caculate_rouge(outputs["reasonings"], gts["reasonings"])
    accuracy = caculate_accuracy(outputs["answers"], gts["answers"])

    evaluation_result = {"bleu-1": bleu_1, "bleu-4": bleu_4, "rouge-L": rouge, "accuracy": accuracy}
    return evaluation_result

def main():
    file_jsonl_path = "/data/TAP/wanglingxiang/wanglingxiang/model/seq_test/ScienceQA_pagerank/1/codellama-7b-test.jsonl"
    completion_dict_row = []
    with open(file_jsonl_path) as file:
        for item in jsonlines.Reader(file):
            completion_dict_row.append(item)

    with open("/data/TAP/wanglingxiang/wanglingxiang/data_sys/TRACE-Benchmark/LLM-CL-Benchmark_500/ScienceQA/test.json", 'r') as f_r:
        data = json.load(f_r)

    assert len(data) == len(completion_dict_row)
    predic = []
    ground = []
    for i, item in enumerate(data):
        assert item['prompt'] == completion_dict_row[i]['prompt']
        predic.append(completion_dict_row[i]['answer'])
        ground.append(item['answer'])
    result = eval(predic, ground)
    print(result)

if __name__ == "__main__":
    main()