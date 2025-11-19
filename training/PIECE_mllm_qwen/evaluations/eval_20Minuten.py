import json
import jsonlines
from metrics import caculate_sari, caculate_bleu, caculate_rouge


def eval(input_sequences, predicted_sequences, ground_truths):
    bleu_1 = caculate_bleu(predicted_sequences, ground_truths, 1)
    bleu_4 = caculate_bleu(predicted_sequences, ground_truths, 4)
    rouge = caculate_rouge(predicted_sequences, ground_truths)
    print("blue_1:", bleu_1, "bleu_4:", bleu_4, "rouge:", rouge)
    sari = caculate_sari(input_sequences, predicted_sequences, ground_truths)
    evaluation_result = {"bleu-1": bleu_1, "bleu-4": bleu_4, "rouge-L": rouge, "sari": sari}
    return evaluation_result

def main():
    file_jsonl_path = "/data/TAP/wanglingxiang/wanglingxiang/model/model_train_data_forget_test_20Minuten/checkpoint-625/test_llama2_7b_base.jsonl"
    completion_dict_row = []
    with open(file_jsonl_path) as file:
        for item in jsonlines.Reader(file):
            completion_dict_row.append(item)

    with open("/data/TAP/wanglingxiang/wanglingxiang/data_sys/TRACE-Benchmark/LLM-CL-Benchmark_5000/20Minuten/test.json", 'r') as f_r:
        data = json.load(f_r)

    assert len(data) == len(completion_dict_row)
    inputs = []
    predic = []
    ground = []
    for i, item in enumerate(data):
        assert item['prompt'] == completion_dict_row[i]['prompt']
        inputs.append(item['prompt'])
        predic.append(completion_dict_row[i]['answer'])
        ground.append(item['answer'])
    result = eval(inputs, predic, ground)
    print(result)

if __name__ == "__main__":
    main()
