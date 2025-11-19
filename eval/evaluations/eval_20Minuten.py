import json
import jsonlines
from metrics import caculate_sari, caculate_bleu, caculate_rouge

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

def eval(input_sequences, predicted_sequences, ground_truths):
    bleu_1 = caculate_bleu(predicted_sequences, ground_truths, 1)
    bleu_4 = caculate_bleu(predicted_sequences, ground_truths, 4)
    rouge = caculate_rouge(predicted_sequences, ground_truths)
    # print("blue_1:", bleu_1, "bleu_4:", bleu_4, "rouge:", rouge)
    # sari = caculate_sari(input_sequences, predicted_sequences, ground_truths)
    # evaluation_result = {"bleu-1": bleu_1, "bleu-4": bleu_4, "rouge-L": rouge, "sari": sari}
    evaluation_result = {"bleu-1": bleu_1, "bleu-4": bleu_4, "rouge-L": rouge}
    return evaluation_result

def main():
    file_jsonl_path = "/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/0924/Fisher_C-STANCE_FOMC_MeetingBank_ScienceQA_NumGLUE-cm_NumGLUE-ds_20Minuten_epoch5_Llama3Exp_0.001/5/20Minuten_multibatch_result.json"
    completion_dict_row = read_json_or_jsonl(file_jsonl_path)

    with open("/data/TAP/wanglingxiang/wanglingxiang/LLM-CL-Benchmark_1000/20Minuten/test.json", 'r') as f_r:
        data = json.load(f_r)

    assert len(data) == len(completion_dict_row)
    inputs = []
    predic = []
    ground = []
    for i, item in enumerate(data):
        assert item['prompt'] == completion_dict_row[i]['input']
        inputs.append(item['prompt'])
        predic.append(completion_dict_row[i]['model_output'])
        ground.append(item['answer'])
    result = eval(inputs, predic, ground)
    print(result)
    
def eval_20Minuten(file_path):
    completion_dict_row = read_json_or_jsonl(file_path)

    with open("/data/TAP/wanglingxiang/wanglingxiang/LLM-CL-Benchmark_1000/20Minuten/test.json", 'r') as f_r:
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
        predic.append(pre)
        ground.append(item['answer'])
    result = eval(inputs, predic, ground)
    # print(result)
    return result

if __name__ == "__main__":
    main()
