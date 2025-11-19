import json
import random
import os
import shutil

for seed in range(40, 50):
    random.seed(seed)
    os.makedirs(f'/home/TAP/TRACE-Benchmark/LLM-CL-Benchmark_500/ScienceQA/100_seed_{seed}', exist_ok=True)
    shutil.copy2('/home/TAP/TRACE-Benchmark/LLM-CL-Benchmark_500/ScienceQA/eval.json',
                 f'/home/TAP/TRACE-Benchmark/LLM-CL-Benchmark_500/ScienceQA/100_seed_{seed}')
    shutil.copy2('/home/TAP/TRACE-Benchmark/LLM-CL-Benchmark_500/ScienceQA/test.json',
                 f'/home/TAP/TRACE-Benchmark/LLM-CL-Benchmark_500/ScienceQA/100_seed_{seed}')
    with open('/home/TAP/TRACE-Benchmark/LLM-CL-Benchmark_500/ScienceQA/sample_100/train.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON 文件内容应为一个列表形式的样本集。")

    random.shuffle(data)

    with open(f'/home/TAP/TRACE-Benchmark/LLM-CL-Benchmark_500/ScienceQA/100_seed_{seed}/train.json', 'w', encoding='utf-8') as f:
        json.dump(data, f)