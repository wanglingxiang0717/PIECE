import json
from metrics import caculate_accuracy
import jsonlines
import re

def eval(predicted_sequences, ground_truths):
    accuracy = caculate_accuracy(predicted_sequences, ground_truths)
    evaluation_result = {"accuracy": accuracy}
    return evaluation_result

def find_first_number(s):
    pattern = r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?'
    match = re.search(pattern, s)
    return match.group(0) if match else None

def main():
    file_jsonl_path = "/data/TAP/wanglingxiang/wanglingxiang/data_sys/TRACE-Benchmark/LLM-CL-Benchmark_500/NumGLUE-cm/test_llama2_7b_base.jsonl"
    completion_dict_row = []
    with open(file_jsonl_path) as file:
        for item in jsonlines.Reader(file):
            completion_dict_row.append(item)

    with open("/data/TAP/wanglingxiang/wanglingxiang/data_sys/TRACE-Benchmark/LLM-CL-Benchmark_500/NumGLUE-cm/test.json", 'r') as f_r:
        data = json.load(f_r)

    assert len(data) == len(completion_dict_row)
    inputs = []
    predic = []
    ground = []
    for i, item in enumerate(data):
        assert item['prompt'] == completion_dict_row[i]['prompt']
        inputs.append(item['prompt'])
        predic.append(find_first_number(completion_dict_row[i]['answer']))
        # print(find_first_number(completion_dict_row[i]['answer']))
        # stop = input()
        ground.append(item['answer'])
    result = eval(predic, ground)
    print(result)

if __name__ == "__main__":
    main()
