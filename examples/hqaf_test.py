import os
import argparse
import json
import copy
import torch
import pandas as pd

from evaluate_model import evaluate,evaluate_question
from init_dataset import init_test_datasets
from init_model import load_model

device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'


def main(params):

    save_dir, batch_size, fusion_type = params["save_dir"], params["bz"], params["fusion_type"].split(",")

    with open(os.path.join(save_dir, "config.json")) as fin:
        config = json.load(fin)
        model_config = copy.deepcopy(config["model_config"])
        for remove_item in ['use_wandb','learning_rate','add_uuid','l2']:
            if remove_item in model_config:
                del model_config[remove_item]    
        trained_params = config["params"]
        fold = trained_params["fold"]
        model_name, dataset_name, emb_type = trained_params["model_name"], trained_params["dataset_name"], trained_params["emb_type"]
        if model_name in ["saint", "sakt", "atdkt"]:
            train_config = config["train_config"]
            seq_len = train_config["seq_len"]
            model_config["seq_len"] = seq_len   

    with open("../configs/data_config.json") as fin:
        curconfig = copy.deepcopy(json.load(fin))
        data_config = curconfig[dataset_name]
        data_config["dataset_name"] = dataset_name

    diff_level = 50
    # test_loader, test_question_loader  = init_test_datasets(data_config, "dimkt", batch_size, diff_level=diff_level)
    test_question_loader  = init_test_datasets(data_config, batch_size, diff_level=diff_level)

    print(f"Start predicting model: {model_name}, embtype: {emb_type}, save_dir: {save_dir}, dataset_name: {dataset_name}")
    print(f"model_config: {model_config}")
    print(f"data_config: {data_config}")

    model = load_model(model_config, data_config, emb_type, save_dir)
    
    #=========================================这是原先保存测试预测文件的代码=========================================
    # save_test_path = os.path.join(save_dir, model.emb_type+"_test_predictions.txt")

    # testauc, testacc = evaluate(model, test_loader, model_name, save_test_path)
    # print(f"testauc: {testauc}, testacc: {testacc}")

    dres = {}  

    q_testaucs, q_testaccs = -1,-1
    if "test_question_file" in data_config and not test_question_loader is None:
        print('into the question test process...')
        save_test_question_path = os.path.join(save_dir, model.emb_type+"_test_question_predictions.txt")
        q_testaucs, q_testaccs = evaluate_question(model, test_question_loader, model_name, fusion_type, save_test_question_path)
        for key in q_testaucs:
            dres["oriauc"+key] = q_testaucs[key]
        for key in q_testaccs:
            dres["oriacc"+key] = q_testaccs[key]
            


    print('q_testaucs=', q_testaucs)
    print('q_testaccs=', q_testaccs)
    print('dres=', dres)
    raw_config = json.load(open(os.path.join(save_dir,"config.json")))
    dres.update(raw_config['params'])
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", type=int, default=64)
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--fusion_type", type=str, default="early_fusion,late_fusion")
    parser.add_argument("--use_wandb", type=int, default=0)

    args = parser.parse_args()
    print(args)
    params = vars(args)
    main(params)
