import re
import collections
import os
import csv

def get_count_and_detection(text, assert_count=False):
    test_pattern = r"Epoch\s(\d+)/(\d+)"
    matches = re.findall(test_pattern, text)
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
        pattern_epoch1 = (
            r"\*{5}\sEvaluating perplexity,\sEpoch\s1/\d+,\sstep\s[\d.]+\s\*{5}\n"
            r"(?:\[.*? loss, ppl\]\s+step:[\d.]+,\s*\tloss:\s*[\d.eE+-]+,\s*\tppl:\s*[\d.eE+-]+\n?)+"
        )
    else:
        pattern_epoch1 = (
            r"\*{5}\sEvaluating perplexity,\sEpoch\s\d+/\d+,\sstep\s[\d.]+\s\*{5}\n"
            r"(?:\[.*? loss, ppl\]\s+step:[\d.]+,\s*\tloss:\s*[\d.eE+-]+,\s*\tppl:\s*[\d.eE+-]+\n?)+"
        )

    matches = re.findall(pattern_epoch1, text)
    name_pattern = r"\[(.*?)\].*?"
    detection_pattern = r"\[(.*?)\]\s+step:([\d.]+),\s*\tloss:\s*([\d.eE+-]+),\s*\tppl:\s*([\d.eE+-]+)"
    save_dict = None

    for match in matches:
        if save_dict is None:
            names = re.findall(name_pattern, match)
            names = [item.split(' ')[0] for item in names]
            save_dict = {item: [] for item in names}

        splits = re.findall(detection_pattern, match)
        for item in splits:
            name = item[0].split(' ')[0]
            step = int(float(item[1]))
            loss = float(item[2])
            ppl = float(item[3])
            save_list = [step, loss, ppl]
            save_dict[name].append(save_list)

    for item in save_dict:
        if epoch_1:
            file_save_name = os.path.join(file_save_path, f"epoch1_{item}.csv")
        else:
            file_save_name = os.path.join(file_save_path, f"epoch_all_{item}.csv")
        with open(file_save_name, "w", newline="", encoding='utf-8') as f_w:
            writer = csv.writer(f_w)
            writer.writerow(['step', 'loss', 'ppl'])
            for save_item in save_dict[item]:
                writer.writerow(save_item)

    print(f"Saved CSVs to {file_save_path}")

if __name__ == "__main__":
    root_dir = "/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/lora_1024"
    train_log_path = os.path.join(root_dir, "train.log")

    if not os.path.exists(train_log_path):
        raise FileNotFoundError(f"{train_log_path} not found!")

    with open(train_log_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 按任务分割日志
    task_splits = re.split(r"\*{5} Task: (.*?) \*{5}", text)
    # re.split 会返回类似 [前面内容, task1, task1内容, task2, task2内容,...]
    # 所以从索引1开始，每隔2取一次
    for i in range(1, len(task_splits), 2):
        task_name = task_splits[i].strip()
        task_text = task_splits[i+1]

        task_dir = os.path.join(root_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        save_epoch(task_dir, task_text, epoch_1=True)
        save_epoch(task_dir, task_text, epoch_1=False)

        print(f"Processed task: {task_name}")
