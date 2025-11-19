import re
import collections
import pandas as pd
import csv
import os

def get_count_and_detection(text, assert_count=False):
    test_pattern = r"""                             
    Epoch\s(\d+)/(\d+)
    """
    matches = re.findall(test_pattern, text)
    # print(len(matches))
    count = collections.defaultdict(int)
    for item in matches:
        assert item[1] == matches[0][1]
        count[item[0]] += 1
    if assert_count:
        for item in count:
            break
        for change in count:
            assert count[change] == count[item]
        print("assert success!")
    return count

def save_epoch(file_save_path, text, epoch_1=True):
    if epoch_1:
        pattern_epoch1 = fr"\*{{5}}\sEvaluating\ perplexity,\sEpoch\s1/\d+,\sstep\s[\d.]+.*?\n" \
                        r"(?:\[.*?loss,\sppl\].*?step:\s[\d.]+\s+loss:\stensor\([\d.eE+-]+,\sdevice='cuda:0'\)\s+ppl:\s[\d.eE+-]+\n?)+"
    else:
        pattern_epoch1 = fr"\*{{5}}\sEvaluating\ perplexity,\sEpoch\s\d+/\d+,\sstep\s[\d.]+.*?\n" \
                        r"(?:\[.*?loss,\sppl\].*?step:\s[\d.]+\s+loss:\stensor\([\d.eE+-]+,\sdevice='cuda:0'\)\s+ppl:\s[\d.eE+-]+\n?)+"        

    matches = re.findall(pattern_epoch1, text)

    name_pattern = r"\[(.*?)\].*?"
    detection_pattern = r"\[(.*?)\].*?step:\s([\d.]+)\s+loss:\stensor\(([\d.eE+-]+),\sdevice='cuda:0'\)\s+ppl:\s([\d.eE+-]+)"

    save_dict = None
    # 打印结果
    for match in matches:
        if save_dict is None:
            names = re.findall(name_pattern, match)
            names = [item.split(' ')[0] for item in names]
            save_dict = {
                item: []
                for item in names
            }

        splits = re.findall(detection_pattern, match)
        for item in splits:
            name = item[0].split(' ')[0]
            step = int(float(item[1]))
            # if item[2] == 'nan':
            #     stop = input(item[2])
            # else:
            loss = float(item[2])
            ppl = float(item[3])
            save_list = [step, loss, ppl]
            save_dict[name].append(save_list)
        # stop = input(save_dict)

    for item in save_dict:
        if epoch_1:
            file_save_name = os.path.join(file_save_path, f"epoch{1}_{item}.csv")
        else:
            file_save_name = os.path.join(file_save_path, f"epoch_all_{item}.csv")
        with open(file_save_name,
                "w", newline="", encoding='utf-8') as f_w:
            writer = csv.writer(f_w)
            writer.writerow(['step', 'loss', 'ppl'])
            save_list = save_dict[item]
            for save_item in save_list:
                writer.writerow(save_item)
    print("finish")

if __name__ == "__main__":
    # for i in range(3):
    file_path = f"/data2/TAP/model/FOMC_001_test_0728_epoch15_random_norm_loss"
    text_path = os.path.join(file_path, 'train.log')
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    file_save_path = os.path.join(file_path, 'result_csv')
    if not os.path.exists(file_save_path):
        os.makedirs(file_save_path)
    save_epoch(file_save_path, text, epoch_1=True)
    save_epoch(file_save_path, text, epoch_1=False)