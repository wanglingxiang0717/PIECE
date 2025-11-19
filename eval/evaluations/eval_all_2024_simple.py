import os
import json
import pandas as pd
from tqdm import tqdm

# 导入你的评测函数
from eval_20Minuten import eval_20Minuten
from eval_CStance import eval_CStance
from eval_FOMC import eval_FOMC
from eval_MeetingBank import eval_MeetingBank
from eval_NumGLUE_cm import eval_NumGLUE_cm
from eval_NumGLUE_ds import eval_NumGLUE_ds
from eval_Py150 import eval_Py150
from eval_ScienceQA import eval_ScienceQA

file_path = "/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/0924/ours_C-STANCE_FOMC_MeetingBank_ScienceQA_NumGLUE-cm_NumGLUE-ds_20Minuten_Py150_epoch5_Llama3Exp_0.001/5/ScienceQA_multibatch_result.json"
result = eval_ScienceQA(file_path)
print(result)