import torch
import numpy as np
import os

from hqaf import hqaf

device = "cpu" if not torch.cuda.is_available() else "cuda"

def init_model(model_config, data_config, emb_type):
    model = hqaf(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    return model

def load_model(model_config, data_config, emb_type, ckpt_path):
    model = init_model(model_config, data_config, emb_type)
    net = torch.load(os.path.join(ckpt_path, emb_type+"_model.ckpt"))
    model.load_state_dict(net)
    return model
