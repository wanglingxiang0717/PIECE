import os
import torch
from tqdm import tqdm

def get_grad_avg(file_path):
    grad_all = 0.0
    shape_all = 0.0
    for file_name in tqdm(os.listdir(file_path)):
        if file_name.endswith('.pt'):
            file_grad_abs = os.path.join(file_path, file_name)
            grad_norm = torch.load(file_grad_abs)
            grad_norm = grad_norm.abs()
            grad_all += grad_norm.sum().item()
            shape_all += len(grad_norm.reshape(-1))
    # stop = input(grad_all / shape_all)
    return grad_all / shape_all

def save_file(file_path, file_save_path):
    grad_avg = get_grad_avg(file_path)
    for file_name in tqdm(os.listdir(file_path)):
        if file_name.endswith('.pt'):
            file_grad_abs = os.path.join(file_path, file_name)
            file_grad_2 = os.path.join(file_path.replace("parameters_grad", "parameters_grad_2"), file_name)
            
            grad_norm = torch.load(file_grad_abs)
            grad_norm = grad_norm.abs()
            grad_norm_2 = torch.load(file_grad_2)
            grad_norm_save = grad_norm.reshape(-1) 
            grad_norm_save_2 = grad_norm_2.reshape(-1)
            # grad_norm_save_new_1 = (grad_norm_save / torch.sqrt(grad_norm_save_2)).reshape(grad_norm.shape)
            grad_norm_save_new_2 = (grad_norm_save / torch.sqrt(grad_norm_save_2 + 1)).reshape(grad_norm.shape)
            # grad_norm_save_new_3 = (grad_norm_save / torch.sqrt(grad_norm_save_2 + grad_avg)).reshape(grad_norm.shape)
            # grad_norm_save_new_4 = (grad_norm_save / (torch.sqrt(grad_norm_save_2) + grad_avg)).reshape(grad_norm.shape)

            # grad_norm_save = grad_norm_save.reshape(grad_norm.shape)
            file_save_new_1 = os.path.join(file_save_path[0], file_name)
            file_save_new_2 = os.path.join(file_save_path[1], file_name)
            file_save_new_3 = os.path.join(file_save_path[2], file_name)
            file_save_new_4 = os.path.join(file_save_path[3], file_name)
            
            # torch.save(grad_norm_save_new_1, file_save_new_1)
            torch.save(grad_norm_save_new_2, file_save_new_2)
            # torch.save(grad_norm_save_new_3, file_save_new_3)
            # torch.save(grad_norm_save_new_4, file_save_new_4)

# target_list = [
#     # "arabic_qa",
#     # "assamese_in",
#     "bangla_bd",
#     "bangla_in",
#     "hindi_in",
#     "turkish_tr",
#     "english_bd",
#     "english_qa",
# ]

# file_path_dict = []
# for target in target_list:
#     file_path_dict.append(f"/data2/TAP/model_exp/{target}_parameters_test_0818_epoch1_random_1000/parameters_grad/epoch0")          

# for file_path in tqdm(file_path_dict):
#     file_save_path_dict = []
#     for i in range(4):
#         file_save_path_dict.append(file_path.replace("parameters_grad", f"parameters_import_new_{i + 1}"))
#     for file_save_path in file_save_path_dict:
#         if not os.path.exists(file_save_path):
#             os.makedirs(file_save_path)
#     save_file(file_path, file_save_path_dict)

def generation_file(args, file_path):
    file_save_path_dict = []
    for i in range(4):
        file_save_path_dict.append(file_path.replace("parameters_grad", f"parameters_import_new_{i + 1}"))
    for file_save_path in file_save_path_dict:
        if not os.path.exists(file_save_path):
            os.makedirs(file_save_path)
    save_file(file_path, file_save_path_dict)

# file_path_dict = [
#     "/data2/TAP/model_exp/ScienceQA_parameters_test_0818_epoch1_random_1000_before/parameters_grad/epoch0"
# ]

# for file_path in tqdm(file_path_dict):
#     file_save_path_dict = []
#     for i in range(4):
#         file_save_path_dict.append(file_path.replace("parameters_grad", f"parameters_import_new_{i + 1}"))
#     for file_save_path in file_save_path_dict:
#         if not os.path.exists(file_save_path):
#             os.makedirs(file_save_path)
#     save_file(file_path, file_save_path_dict)