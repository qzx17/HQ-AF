import os, sys
import json

from torch.utils.data import DataLoader
import numpy as np
from hqaf_dataloader import hqafDataset



def init_test_datasets(data_config, batch_size, diff_level=None):
    dataset_name = data_config["dataset_name"]
    test_question_loader, test_question_window_loader = None, None
    test_question_dataset = hqafDataset(data_config["dpath"],os.path.join(data_config["dpath"], data_config["test_question_file"]), data_config["input_type"], {-1}, True, diff_level=diff_level)

    if "test_question_file" in data_config:
        print(f"has test_question_file!")
        test_question_loader,test_question_window_loader = None,None
        if not test_question_dataset is None:
            test_question_loader = DataLoader(test_question_dataset, batch_size=batch_size, shuffle=False)

    return test_question_loader

def update_gap(max_rgap, max_sgap, max_pcount, cur):
    max_rgap = cur.max_rgap if cur.max_rgap > max_rgap else max_rgap
    max_sgap = cur.max_sgap if cur.max_sgap > max_sgap else max_sgap
    max_pcount = cur.max_pcount if cur.max_pcount > max_pcount else max_pcount
    return max_rgap, max_sgap, max_pcount

def init_dataset4train(dataset_name, data_config, i, batch_size, diff_level=None):
    print(f"dataset_name:{dataset_name}")
    data_config = data_config[dataset_name]
    all_folds = set(data_config["folds"])
    
    curvalid = hqafDataset(data_config["dpath"],os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], {i}, diff_level=diff_level)
    curtrain = hqafDataset(data_config["dpath"],os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], all_folds - {i}, diff_level=diff_level)
   
    train_loader = DataLoader(curtrain, batch_size=batch_size)
    valid_loader = DataLoader(curvalid, batch_size=batch_size)
    
 
    return train_loader, valid_loader#, test_loader, test_window_loader
