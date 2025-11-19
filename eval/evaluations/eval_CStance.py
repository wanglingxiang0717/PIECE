import json
import jsonlines
from metrics import caculate_accuracy

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

def eval(predicted_sequences, ground_truths):
    accuracy = caculate_accuracy(predicted_sequences, ground_truths)
    evaluation_result = {"accuracy": accuracy}
    return evaluation_result

def find_first_occurrence(s):
    # target = {'A', 'B', 'C', 'D'}
    # return next((char for char in s if char in target), None)
    target = {'A', 'B', 'C', 'D'}
    for char in s[:4]:
        if char in target:
            return char
    if len(s) >= 2:
        first_two_chars = s[:2]
        if first_two_chars == "支持":
            return "A"
        elif first_two_chars == "反对":
            return "B"
        elif first_two_chars == "中立":
            return "C"
    return None

def main():
    file_jsonl_path = "/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/lora_1024/C-STANCE/0/C-STANCE_multibatch_result.json"
    completion_dict_row = read_json_or_jsonl(file_jsonl_path)

    with open("/data/TAP/wanglingxiang/wanglingxiang/LLM-CL-Benchmark_1000/C-STANCE/test.json", 'r') as f_r:
        data = json.load(f_r)

    assert len(data) == len(completion_dict_row)
    inputs = []
    predic = []
    ground = []
    for i, item in enumerate(data):
        assert item['prompt'] == completion_dict_row[i]['input']
        inputs.append(item['prompt'])
        predic.append(find_first_occurrence(completion_dict_row[i]['model_output']))
        ground.append(item['answer'])
    result = eval(predic, ground)
    print(result)
    
def eval_CStance(file_path):
    completion_dict_row = read_json_or_jsonl(file_path)

    with open("/data/TAP/wanglingxiang/wanglingxiang/LLM-CL-Benchmark_1000/C-STANCE/test.json", 'r') as f_r:
        data = json.load(f_r)

    assert len(data) == len(completion_dict_row)
    inputs = []
    predic = []
    ground = []
    for i, item in enumerate(data):
        assert item['prompt'] == completion_dict_row[i]['input']
        inputs.append(item['prompt'])
        if '</think>' in completion_dict_row[i]['model_output']:
            pre = completion_dict_row[i]['model_output'].split('</think>')[-1]
        else:
            pre = completion_dict_row[i]['model_output']
        predic.append(find_first_occurrence(pre))
        ground.append(item['answer'])
    result = eval(predic, ground)
    # print(result)
    return result

if __name__ == "__main__":
    main()