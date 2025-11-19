import re
import json
from metrics import caculate_fuzz

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

def postprocess(code):
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    lits = re.findall(pattern, code)
    for lit in lits:
        code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
    return code


def eval(predicted_sequences, ground_truths):
    outputs = []
    for output in predicted_sequences:
        outputs.append(postprocess(output))
    gts = []
    for gt in ground_truths:
        gts.append(postprocess(gt))

    fuzz = caculate_fuzz(outputs, gts)
    evaluation_result = {"similarity": fuzz}
    
    return evaluation_result

def main():
    file_jsonl_path = "/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/0924/Fisher_C-STANCE_FOMC_MeetingBank_ScienceQA_NumGLUE-cm_NumGLUE-ds_20Minuten_Py150_epoch5_Llama3Exp_0.001/5/Py150_multibatch_result.json"
    completion_dict_row = read_json_or_jsonl(file_jsonl_path)

    with open("/data/TAP/wanglingxiang/wanglingxiang/LLM-CL-Benchmark_1000/Py150/test.json", 'r') as f_r:
        data = json.load(f_r)

    assert len(data) == len(completion_dict_row)
    inputs = []
    predic = []
    ground = []
    for i, item in enumerate(data):
        assert item['prompt'] == completion_dict_row[i]['input']
        inputs.append(item['prompt'])
        # stop = input(completion_dict_row[i]['model_output'].split('<EOL>')[-1])
        predic.append(completion_dict_row[i]['model_output'].split('<EOL>')[-1])
        ground.append(item['answer'])
    result = eval(predic, ground)
    print(result)
    
def eval_Py150(file_jsonl_path):
    # file_jsonl_path = "/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/0924/Fisher_C-STANCE_FOMC_MeetingBank_ScienceQA_NumGLUE-cm_NumGLUE-ds_20Minuten_Py150_epoch5_Llama3Exp_0.001/5/Py150_multibatch_result.json"
    completion_dict_row = read_json_or_jsonl(file_jsonl_path)

    with open("/data/TAP/wanglingxiang/wanglingxiang/LLM-CL-Benchmark_1000/Py150/test.json", 'r') as f_r:
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
        # stop = input(completion_dict_row[i]['model_output'].split('<EOL>')[-1])
        predic.append(pre.split('<EOL>')[-1])
        ground.append(item['answer'])
    result = eval(predic, ground)
    # print(result)
    return result
    
if __name__ == "__main__":
    main()

