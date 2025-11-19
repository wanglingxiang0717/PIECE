import json
import jsonlines
from metrics import caculate_accuracy


def eval(predicted_sequences, ground_truths):
    accuracy = caculate_accuracy(predicted_sequences, ground_truths)
    evaluation_result = {"accuracy": accuracy}
    return evaluation_result

def find_first_occurrence(s):
    target = {'A', 'B', 'C', 'D'}
    return next((char for char in s if char in target), None)

def main():
    file_jsonl_path = "/data2/TAP/model_exp/model_save/Meta-Llama-3-8B-Instruct_FOMC_0909_epoch5_full_params/1/FOMC_test.jsonl"
    completion_dict_row = []
    with open(file_jsonl_path) as file:
        for item in jsonlines.Reader(file):
            completion_dict_row.append(item)

    with open("/home/TAP/TRACE-Benchmark/LLM-CL-Benchmark_1000/FOMC/test.json", 'r') as f_r:
        data = json.load(f_r)

    assert len(data) == len(completion_dict_row)
    inputs = []
    predic = []
    ground = []
    for i, item in enumerate(data):
        assert item['prompt'] == completion_dict_row[i]['prompt']
        inputs.append(item['prompt'])
        predic.append(find_first_occurrence(completion_dict_row[i]['answer']))
        ground.append(item['answer'])
    result = eval(predic, ground)
    print(result)

if __name__ == "__main__":
    main()