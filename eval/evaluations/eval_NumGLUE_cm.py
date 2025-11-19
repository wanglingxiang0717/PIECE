import json
from metrics import caculate_accuracy
import jsonlines
import re

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

def find_first_number(s):
    pattern = r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?'
    match = re.search(pattern, s)
    return match.group(0) if match else None

def main():
    file_jsonl_path = "/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/0924/Fisher_C-STANCE_FOMC_MeetingBank_ScienceQA_NumGLUE-cm_NumGLUE-ds_20Minuten_Py150_epoch5_Llama3Exp_0.001/5/NumGLUE-cm_multibatch_result.json"
    completion_dict_row = completion_dict_row = read_json_or_jsonl(file_jsonl_path)

    with open("/data/TAP/wanglingxiang/wanglingxiang/LLM-CL-Benchmark_1000/NumGLUE-cm/test.json", 'r') as f_r:
        data = json.load(f_r)

    assert len(data) == len(completion_dict_row)
    inputs = []
    predic = []
    ground = []
    for i, item in enumerate(data):
        assert item['prompt'] == completion_dict_row[i]['input']
        inputs.append(item['prompt'])
        predic.append(find_first_number(completion_dict_row[i]['model_output']))
        # print(find_first_number(completion_dict_row[i]['answer']))
        # stop = input()
        ground.append(item['answer'])
    result = eval(predic, ground)
    print(result)
    
def eval_NumGLUE_cm(file_jsonl_path):
    completion_dict_row = completion_dict_row = read_json_or_jsonl(file_jsonl_path)

    with open("/data/TAP/wanglingxiang/wanglingxiang/LLM-CL-Benchmark_1000/NumGLUE-cm/test.json", 'r') as f_r:
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
        predic.append(find_first_number(pre))
        # print(find_first_number(completion_dict_row[i]['answer']))
        # stop = input()
        ground.append(item['answer'])
    result = eval(predic, ground)
    # print(result)
    return result
    

if __name__ == "__main__":
    main()