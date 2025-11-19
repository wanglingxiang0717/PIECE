import sys
sys.dont_write_bytecode = True

import argparse
import os
import math
import sys
from tqdm import tqdm
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import shutil
import random
import subprocess
import torch
from param_protection.get_parameters_import_files import generation_file
from param_protection.extract_and_save_new import generation_mask_file_simple


# generation_file(None, file_path=f"/data2/TAP/model_exp/0920_20Minuten_ours_parameters_test_epoch1_random_1000/parameters_grad/epoch0")
generation_mask_file_simple(top_ratio=0.001, grad_dir="/data2/TAP/model_exp/0920_20Minuten_ours_parameters_test_epoch1_random_1000/parameters_import_new_2/epoch0")