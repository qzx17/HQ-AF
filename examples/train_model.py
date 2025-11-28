import os, sys
import torch
import torch.nn as nn
from torch.nn.functional import one_hot, binary_cross_entropy, cross_entropy
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from evaluate_model import evaluate
from torch.autograd import Variable, grad
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cal_loss(model, ys, r, rshft, sm):
    model_name = model.model_name

    if model_name in ["hqaf"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double())
  
    
    return loss


def model_forward(model, data, rel=None):
    model_name = model.model_name
    # if model_name in ["dkt_forget", "lpkt"]:
    #     q, c, r, qshft, cshft, rshft, m, sm, d, dshft = data
    if model_name in ["dkt_forget", "bakt_time"]:
        dcur, dgaps = data
    else:
        dcur = data
    #----------------这里是改动----------------
    q, c, r, t,sd,qd, qu = dcur["qseqs"].to(device), dcur["cseqs"].to(device), dcur["rseqs"].to(device), dcur["tseqs"].to(device),dcur["sdseqs"].to(device),dcur["qdseqs"].to(device), dcur["quseqs"].to(device)
    qshft, cshft, rshft, tshft,sdshft,qdshft, qushft = dcur["shft_qseqs"].to(device), dcur["shft_cseqs"].to(device), dcur["shft_rseqs"].to(device), dcur["shft_tseqs"].to(device),dcur["shft_sdseqs"].to(device),dcur["shft_qdseqs"].to(device), dcur["shft_quseqs"].to(device)
    utT,pT = dcur["utTseqs"].to(device), dcur["pTseqs"].to(device)
    utTshft, pTshft = dcur["shft_utTseqs"].to(device), dcur["shft_pTseqs"].to(device)

    m, sm = dcur["masks"].to(device), dcur["smasks"].to(device)

    ys, preloss = [], []
    cq = torch.cat((q[:,0:1], qshft), dim=1)
    cc = torch.cat((c[:,0:1], cshft), dim=1)
    cr = torch.cat((r[:,0:1], rshft), dim=1)
    # cut = torch.cat((ut[:,0:1], utshft), dim=1)
    cutT = torch.cat((utT[:,0:1], utTshft), dim=1)
    cpT = torch.cat((pT[:,0:1], pTshft), dim=1)
    

    #----------------这里是改动----------------
    csd = torch.cat((sd[:,0:1], sdshft), dim=1)
    cqd = torch.cat((qd[:,0:1], qdshft), dim=1)
    cqu = torch.cat((qu[:,0:1], qushft), dim=1)


    if model_name in ["hqaf"]:               
        y, d_loss = model(cc.long(), cr.long(), cq.long(), cd = csd.long(), qd = cqd.long(), qu = cqu.long(), cutT = cutT.long(), cpT = cpT.long(), sdshft = sdshft.long(), sm = sm)
        # y, reg_loss = model(cc.long(), cr.long(), cq.long(), cd = csd.long(), qd = cqd.long())
        ys.append(y[:,1:])
    

    # loss = cal_loss(model, ys, r, rshft, sm)
    # loss = cal_loss(model, ys, r, rshft, sm) + d_loss * 0.01
    loss = cal_loss(model, ys, r, rshft, sm)
    return loss 
    

def train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, test_loader=None, test_window_loader=None, save_model=False, data_config=None, fold=None):
    max_auc, best_epoch = 0, -1
    train_step = 0

    rel = None
    if model.model_name == "rkt":
        dpath = data_config["dpath"]
        dataset_name = dpath.split("/")[-1]
        tmp_folds = set(data_config["folds"]) - {fold}
        folds_str = "_" + "_".join([str(_) for _ in tmp_folds])
        if dataset_name in ["algebra2005", "bridge2algebra2006"]:
            fname = "phi_dict" + folds_str + ".pkl"
            rel = pd.read_pickle(os.path.join(dpath, fname))
        else:
            fname = "phi_array" + folds_str + ".pkl" 
            rel = pd.read_pickle(os.path.join(dpath, fname))

    if model.model_name=='lpkt':
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 10, gamma=0.5)
    # set min learning rate
    min_learning_rate = 1e-4
    # set scheduler to update the lr.
    scheduler = torch.optim.lr_scheduler.StepLR(opt, 5, gamma=0.5)
    for i in range(1, num_epochs + 1):
        loss_mean = []
        for data in train_loader:
            train_step+=1
            model.train()
            if model.model_name=='rkt':
                loss = model_forward(model, data, rel)
            else:
                loss = model_forward(model, data)
            opt.zero_grad()
            loss.backward()#compute gradients
            if model.model_name == "rkt":
                clip_grad_norm_(model.parameters(), model.grad_clip)
            opt.step()#update model’s parameters
                

        
            loss_mean.append(loss.detach().cpu().numpy())
        if model.model_name=='lpkt':
            scheduler.step()#update each epoch
        # 更新学习率，但在达到最小阈值时停止更新
        current_lr = opt.param_groups[0]['lr']
        if current_lr > min_learning_rate:
            scheduler.step()
        print(f"Epoch {i}, Learning Rate: {scheduler.get_last_lr()}")
        loss_mean = np.mean(loss_mean)
        
        if model.model_name=='rkt':
            auc, acc = evaluate(model, valid_loader, model.model_name, rel)
        else:
            auc, acc = evaluate(model, valid_loader, model.model_name)
        ### atkt 有diff， 以下代码导致的
        ### auc, acc = round(auc, 4), round(acc, 4)

        if auc > max_auc+1e-3:
            if save_model:
                #保存模型
                torch.save(model.state_dict(), os.path.join(ckpt_path, model.emb_type+"_model.ckpt"))
            max_auc = auc
            best_epoch = i
            testauc, testacc = -1, -1
            window_testauc, window_testacc = -1, -1
            if not save_model:
                if test_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_predictions.txt")
                    testauc, testacc = evaluate(model, test_loader, model.model_name, save_test_path)
                if test_window_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_window_predictions.txt")
                    window_testauc, window_testacc = evaluate(model, test_window_loader, model.model_name, save_test_path)
            validauc, validacc = auc, acc
        print(f"Epoch: {i}, validauc: {validauc:.4}, validacc: {validacc:.4}, best epoch: {best_epoch}, best auc: {max_auc:.4}, train loss: {loss_mean}, emb_type: {model.emb_type}, model: {model.model_name}, save_dir: {ckpt_path}")
        print(f"            testauc: {round(testauc,4)}, testacc: {round(testacc,4)}, window_testauc: {round(window_testauc,4)}, window_testacc: {round(window_testacc,4)}")


        if i - best_epoch >= 10:
            break
    return testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch
