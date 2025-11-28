import numpy as np
import torch
from torch import nn
from torch.nn.functional import one_hot
from sklearn import metrics
import pandas as pd
import csv

device = "cpu" if not torch.cuda.is_available() else "cuda"

def save_cur_predict_result(dres, q, r, d, t, m, sm, p):
    # dres, q, r, qshft, rshft, m, sm, y
    results = []
    for i in range(0, t.shape[0]):
        cps = torch.masked_select(p[i], sm[i]).detach().cpu()
        cts = torch.masked_select(t[i], sm[i]).detach().cpu()
    
        cqs = torch.masked_select(q[i], m[i]).detach().cpu()
        crs = torch.masked_select(r[i], m[i]).detach().cpu()

        cds = torch.masked_select(d[i], sm[i]).detach().cpu()

        qs, rs, ts, ps, ds = [], [], [], [], []
        for cq, cr in zip(cqs.int(), crs.int()):
            qs.append(cq.item())
            rs.append(cr.item())
        for ct, cp, cd in zip(cts.int(), cps, cds.int()):
            ts.append(ct.item())
            ps.append(cp.item())
            ds.append(cd.item())
        try:
            auc = metrics.roc_auc_score(
                y_true=np.array(ts), y_score=np.array(ps)
            )
            
        except Exception as e:
            # print(e)
            auc = -1
        prelabels = [1 if p >= 0.5 else 0 for p in ps]
        acc = metrics.accuracy_score(ts, prelabels)
        dres[len(dres)] = [qs, rs, ds, ts, ps, prelabels, auc, acc]
        results.append(str([qs, rs, ds, ts, ps, prelabels, auc, acc]))
    return "\n".join(results)


def evaluate(model, test_loader, model_name, rel=None, save_path=""):
    if save_path != "":
        fout = open(save_path, "w", encoding="utf8")
    with torch.no_grad():
        y_trues = []
        y_scores = []
        dres = dict()
        test_mini_index = 0
        for data in test_loader:
            # if model_name in ["dkt_forget", "lpkt"]:
            #     q, c, r, qshft, cshft, rshft, m, sm, d, dshft = data
            dcur = data
            q, c, r, sd,qd, qu = dcur["qseqs"], dcur["cseqs"], dcur["rseqs"], dcur["sdseqs"],dcur["qdseqs"], dcur["quseqs"].to(device)
            qshft, cshft, rshft, sdshft,qdshft,qushft = dcur["shft_qseqs"], dcur["shft_cseqs"], dcur["shft_rseqs"], dcur["shft_sdseqs"],dcur["shft_qdseqs"],dcur["shft_quseqs"]
            sd, qd, qu, sdshft, qdshft, qushft = sd.to(device), qd.to(device), qu.to(device), sdshft.to(device), qdshft.to(device), qushft.to(device)
            utT,pT = dcur["utTseqs"].to(device), dcur["pTseqs"].to(device)
            utTshft, pTshft = dcur["shft_utTseqs"].to(device), dcur["shft_pTseqs"].to(device)

            m, sm = dcur["masks"], dcur["smasks"]
            q, c, r, qshft, cshft, rshft, m, sm = q.to(device), c.to(device), r.to(device), qshft.to(device), cshft.to(device), rshft.to(device), m.to(device), sm.to(device)
            model.eval()

            # print(f"before y: {y.shape}")
            cq = torch.cat((q[:,0:1], qshft), dim=1)
            cc = torch.cat((c[:,0:1], cshft), dim=1)
            cr = torch.cat((r[:,0:1], rshft), dim=1)
            csd = torch.cat((sd[:,0:1], sdshft), dim=1)
            cqd = torch.cat((qd[:,0:1], qdshft), dim=1)
            cqu = torch.cat((qu[:,0:1], qushft), dim=1)
            cutT = torch.cat((utT[:,0:1], utTshft), dim=1)
            cpT = torch.cat((pT[:,0:1], pTshft), dim=1)

            y, _ = model(cc.long(), cr.long(), cq.long(), cd = csd.long(), qd = cqd.long(), qu = cqu.long(), cutT=cutT.long(), cpT = cpT.long(), sdshft = sdshft.long(), sm = sm)
            # y, reg_loss = model(cc.long(), cr.long(), cq.long(), cd = csd.long(), qd = cqd.long())
            y = y[:,1:]
            # print(f"after y: {y.shape}")
            # save predict result
            if save_path != "":
                result = save_cur_predict_result(dres, c, r, cshft, rshft, m, sm, y)
                fout.write(result+"\n")

            y = torch.masked_select(y, sm).detach().cpu()
            # print(f"pred_results:{y}")  
            t = torch.masked_select(rshft, sm).detach().cpu()

            y_trues.append(t.numpy())
            y_scores.append(y.numpy())
            test_mini_index+=1
        ts = np.concatenate(y_trues, axis=0)
        ps = np.concatenate(y_scores, axis=0)
        print(f"ts.shape: {ts.shape}, ps.shape: {ps.shape}")
        auc = metrics.roc_auc_score(y_true=ts, y_score=ps)

        prelabels = [1 if p >= 0.5 else 0 for p in ps]
        acc = metrics.accuracy_score(ts, prelabels)
    # if save_path != "":
    #     pd.to_pickle(dres, save_path+".pkl")
    return auc, acc

def early_fusion(curhs, model, model_name):
    
    output = model.out(curhs[0]).squeeze(-1)
    m = nn.Sigmoid()
    p = m(output)
    return p

def late_fusion(dcur, curdf, fusion_type=["mean", "vote", "all"]):
    high, low = [], []
    for pred in curdf["preds"]:
        if pred >= 0.5:
            high.append(pred)
        else:
            low.append(pred)

    if "mean" in fusion_type:
        dcur.setdefault("late_mean", [])
        dcur["late_mean"].append(round(float(curdf["preds"].mean()), 4))
    if "vote" in fusion_type:
        dcur.setdefault("late_vote", [])
        correctnum = list(curdf["preds"]>=0.5).count(True)
        late_vote = np.mean(high) if correctnum / len(curdf["preds"]) >= 0.5 else np.mean(low)
        dcur["late_vote"].append(late_vote)
    if "all" in fusion_type:
        dcur.setdefault("late_all", [])
        late_all = np.mean(high) if correctnum == len(curdf["preds"]) else np.mean(low)
        dcur["late_all"].append(late_all)
    return 

def effective_fusion(df, model, model_name, fusion_type):
    dres = dict()
    df = df.groupby("qidx", as_index=True, sort=True)#.mean()

    curhs, curr = [[], []], []
    dcur = {"late_trues": [], "qidxs": [], "questions": [], "concepts": [], "row": [], "concept_preds": []}
    for ui in df:
        curdf = ui[1]
        curhs[0].append(curdf["hidden"].mean().astype(float))
        
        curr.append(int(curdf["response"].mean()))
        dcur["late_trues"].append(int(curdf["response"].mean()))
        dcur["qidxs"].append(ui[0])
        dcur["row"].append(int(curdf["row"].mean()))
        dcur["questions"].append(",".join([str(int(s)) for s in curdf["questions"].tolist()]))
        dcur["concepts"].append(",".join([str(int(s)) for s in curdf["concepts"].tolist()]))
        late_fusion(dcur, curdf)
        # save original predres in concepts
        dcur["concept_preds"].append(",".join([str(round(s, 4)) for s in (curdf["preds"].tolist())]))

    for key in dcur:
        dres.setdefault(key, [])
        dres[key].append(np.array(dcur[key]))
    # early fusion
    curhs = [torch.tensor(curh).float().to(device) for curh in curhs]
    curr = torch.tensor(curr).long().to(device)
    p = early_fusion(curhs, model, model_name)
    dres.setdefault("early_trues", [])
    dres["early_trues"].append(curr.cpu().numpy())
    dres.setdefault("early_preds", [])
    dres["early_preds"].append(p.cpu().numpy())
    return dres

def group_fusion(dmerge, model, model_name, fusion_type, fout):
    hs, sms, cq, cc, rs, ps, qidxs, rests, orirows = dmerge["hs"], dmerge["sm"], dmerge["cq"], dmerge["cc"], dmerge["cr"], dmerge["y"], dmerge["qidxs"], dmerge["rests"], dmerge["orirow"]
    if cq.shape[1] == 0:
        cq = cc
    
    alldfs, drest = [], dict() # not predict infos!
    # print(f"real bz in group fusion: {rs.shape[0]}")
    realbz = rs.shape[0]
    for bz in range(rs.shape[0]):
        cursm = ([0] + sms[bz].cpu().tolist())
        curqidxs = ([-1] + qidxs[bz].cpu().tolist())
        currests = ([-1] + rests[bz].cpu().tolist())
        currows = ([-1] + orirows[bz].cpu().tolist())
        curps = ([-1] + ps[bz].cpu().tolist())
        # print(f"qid: {len(curqidxs)}, select: {len(cursm)}, response: {len(rs[bz].cpu().tolist())}, preds: {len(curps)}")
        df = pd.DataFrame({"qidx": curqidxs, "rest": currests, "row": currows, "select": cursm, 
                "questions": cq[bz].cpu().tolist(), "concepts": cc[bz].cpu().tolist(), "response": rs[bz].cpu().tolist(), "preds": curps})
   
        df["hidden"] = [np.array(a) for a in hs[0][bz].cpu().tolist()]
        df = df[df["select"] != 0]
        alldfs.append(df)
    
    effective_dfs, rest_start = [], -1
    flag = False
    for i in range(len(alldfs) - 1, -1, -1):
        df = alldfs[i]
        counts = (df["rest"] == 0).value_counts()
        if not flag and False not in counts: # has no question rest > 0
            flag =True
            effective_dfs.append(df)
            rest_start = i + 1
        elif flag:
            effective_dfs.append(df)
    if rest_start == -1:
        rest_start = 0
    # merge rest
    for key in dmerge.keys():
        if key == "hs":
            drest[key] = []
            drest[key] = [dmerge[key][0][rest_start:]]
        else:
            drest[key] = dmerge[key][rest_start:] 
    restlen = drest["cr"].shape[0]

    dfs = dict()
    for df in effective_dfs:
        for i, row in df.iterrows():
            for key in row.keys():
                dfs.setdefault(key, [])
                dfs[key].extend([row[key]])
    df = pd.DataFrame(dfs)

    if df.shape[0] == 0:
        return {}, drest

    dres = effective_fusion(df, model, model_name, fusion_type)
            
    dfinal = dict()
    for key in dres:
        dfinal[key] = np.concatenate(dres[key], axis=0)
    early = True
    save_question_res(dfinal, fout, early)
    return dfinal, drest

def save_question_res(dres, fout, early=False):
    # print(f"dres: {dres.keys()}")
    # qidxs, late_trues, late_mean, late_vote, late_all, early_trues, early_preds
    for i in range(0, len(dres["qidxs"])):
        row, qidx, qs, cs, lt, lm, lv, la = dres["row"][i], dres["qidxs"][i], dres["questions"][i], dres["concepts"][i], \
            dres["late_trues"][i], dres["late_mean"][i], dres["late_vote"][i], dres["late_all"][i]
        conceptps = dres["concept_preds"][i]
        curres = [row, qidx, qs, cs, conceptps, lt, lm, lv, la]
        if early:
            et, ep = dres["early_trues"][i], dres["early_preds"][i]
            curres = curres + [et, ep]
        curstr = "\t".join([str(round(s, 4)) if type(s) == type(0.1) or type(s) == np.float32 else str(s) for s in curres])
        fout.write(curstr + "\n")

def evaluate_question(model, test_loader, model_name, fusion_type=["early_fusion", "late_fusion"], save_path=""):
    # dkt / dkt+ / dkt_forget / atkt: give past -> predict all. has no early fusion!!!
    # dkvmn / hqaf / saint: give cur -> predict cur
    # sakt: give past+cur -> predict cur
    # kqn: give past+cur -> predict cur
    if save_path != "":
        fout = open(save_path, "w", encoding="utf8")
        fout.write("\t".join(["orirow", "qidx", "questions", "concepts", "concept_preds", "late_trues", "late_mean", "late_vote", "late_all", "early_trues", "early_preds"]) + "\n")
        
    with torch.no_grad():
        dinfos = dict()
        dhistory = dict()
        history_keys = ["hs", "sm", "cq", "cc", "cr", "y", "qidxs", "rests", "orirow"]
        # for key in history_keys:
        #     dhistory[key] = []
        y_trues, y_scores = [], []
        lenc = 0
        for data in test_loader:
            if model_name in ["dkt_forget", "bakt_time"]:
                dcurori, dgaps, dqtest = data
            else:
                dcurori, dqtest = data

            q, c, r ,sd, qd, qu = dcurori["qseqs"], dcurori["cseqs"], dcurori["rseqs"],dcurori["sdseqs"],dcurori["qdseqs"],dcurori["quseqs"]
            qshft, cshft, rshft, sdshft, qdshft, qushft = dcurori["shft_qseqs"], dcurori["shft_cseqs"], dcurori["shft_rseqs"], dcurori["shft_sdseqs"],dcurori["shft_qdseqs"],dcurori["shft_quseqs"]
            sd, qd, qu, sdshft, qdshft, qushft = sd.to(device), qd.to(device), qu.to(device), sdshft.to(device), qdshft.to(device), qushft.to(device)
            utT,pT = dcurori["utTseqs"].to(device), dcurori["pTseqs"].to(device)
            utTshft, pTshft = dcurori["shft_utTseqs"].to(device), dcurori["shft_pTseqs"].to(device)

            m, sm = dcurori["masks"], dcurori["smasks"]
            q, c, r, qshft, cshft, rshft, m, sm = q.to(device), c.to(device), r.to(device), qshft.to(device), cshft.to(device), rshft.to(device), m.to(device), sm.to(device)
            qidxs, rests, orirow = dqtest["qidxs"], dqtest["rests"], dqtest["orirow"]
            lenc += q.shape[0]
            # print("="*20)
            # print(f"start predict seqlen: {lenc}")
            model.eval()

            # print(f"before y: {y.shape}")
            cq = torch.cat((q[:,0:1], qshft), dim=1)
            cc = torch.cat((c[:,0:1], cshft), dim=1)
            cr = torch.cat((r[:,0:1], rshft), dim=1)

            #----------------这里是改动----------------
            csd = torch.cat((sd[:,0:1], sdshft), dim=1)
            cqd = torch.cat((qd[:,0:1], qdshft), dim=1)
            cqu = torch.cat((qu[:,0:1], qushft), dim=1)
            cutT = torch.cat((utT[:,0:1], utTshft), dim=1)
            cpT = torch.cat((pT[:,0:1], pTshft), dim=1)

            dcur = dict()
           
            y, h = model(cc.long(), cr.long(), cq.long(), True, cd = csd.long(), qd = cqd.long(), qu = cqu.long(), cutT = cutT.long(), cpT = cpT.long(), sdshft = sdshft.long(), sm = sm)
            y = y[:,1:]
           
            concepty = torch.masked_select(y, sm).detach().cpu()
            conceptt = torch.masked_select(rshft, sm).detach().cpu()

            y_trues.append(conceptt.numpy())
            y_scores.append(concepty.numpy())

            # hs, sms, rs, ps, qidxs, model, model_name, fusion_type
            hs = []
            hs = [h]
            dcur["hs"], dcur["sm"], dcur["cq"], dcur["cc"], dcur["cr"], dcur["y"], dcur["qidxs"], dcur["rests"], dcur["orirow"] = hs, sm, cq, cc, cr, y, qidxs, rests, orirow
            # merge history
            dmerge = dict()
            for key in history_keys:
                if len(dhistory) == 0:
                    dmerge[key] = dcur[key]
                else:
                    if key == "hs":
                        dmerge[key] = []
                        
                        dmerge[key] = [torch.cat((dhistory[key][0], dcur[key][0]), dim=0)]                            
                    else:
                        dmerge[key] = torch.cat((dhistory[key], dcur[key]), dim=0)
                
            dcur, dhistory = group_fusion(dmerge, model, model_name, fusion_type, fout)
            for key in dcur:
                dinfos.setdefault(key, [])
                dinfos[key].append(dcur[key])

            if "early_fusion" in dinfos and "late_fusion" in dinfos:
                assert dinfos["early_trues"][-1].all() == dinfos["late_trues"][-1].all()
            # import sys
            # sys.exit()
        # ori concept eval
        aucs, accs = dict(), dict()
        ts = np.concatenate(y_trues, axis=0)
        ps = np.concatenate(y_scores, axis=0)
        # print(f"ts.shape: {ts.shape}, ps.shape: {ps.shape}")
        auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
        prelabels = [1 if p >= 0.5 else 0 for p in ps]
        acc = metrics.accuracy_score(ts, prelabels)
        aucs["concepts"] = auc
        accs["concepts"] = acc

        # print(f"dinfos: {dinfos.keys()}")
        for key in dinfos:
            if key not in ["late_mean", "late_vote", "late_all", "early_preds"]:
                continue
            ts = np.concatenate(dinfos['late_trues'], axis=0) # early_trues == late_trues
            ps = np.concatenate(dinfos[key], axis=0)
            # print(f"key: {key}, ts.shape: {ts.shape}, ps.shape: {ps.shape}")
            auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
            prelabels = [1 if p >= 0.5 else 0 for p in ps]
            acc = metrics.accuracy_score(ts, prelabels)
            aucs[key] = auc
            accs[key] = acc
    return aucs, accs


def log2(t):
    import math
    return round(math.log(t+1, 2))

def calC(row, data_config):
    repeated_gap, sequence_gap, past_counts = [], [], []
    uid = row["uid"]
    # default: concepts
    skills = row["concepts"].split(",")
    timestamps = row["timestamps"].split(",")
    dlastskill, dcount = dict(), dict()
    pret = None
    idx = -1
    for s, t in zip(skills, timestamps):
        idx += 1
        s, t = int(s), int(t)
        if s not in dlastskill or s == -1:
            curRepeatedGap = 0
        else:
            curRepeatedGap = log2((t - dlastskill[s]) / 1000 / 60) + 1 # minutes
        dlastskill[s] = t

        repeated_gap.append(curRepeatedGap)
        if pret == None or t == -1:
            curLastGap = 0
        else:
            curLastGap = log2((t - pret) / 1000 / 60) + 1
        pret = t
        sequence_gap.append(curLastGap)

        dcount.setdefault(s, 0)
        ccount = log2(dcount[s])
        ccount = data_config["num_pcount"] - 1 if ccount >= data_config["num_pcount"] else ccount
        past_counts.append(ccount)
        
        dcount[s] += 1
    return repeated_gap, sequence_gap, past_counts           

def get_info_dkt_forget(row, data_config):
    dforget = dict()
    rgap, sgap, pcount = calC(row, data_config)

    ## TODO
    dforget["rgaps"], dforget["sgaps"], dforget["pcounts"] = rgap, sgap, pcount
    return dforget

def evaluate_splitpred_question(model, data_config, testf, model_name, save_path="", use_pred=False, train_ratio=0.2, atkt_pad=False):
    if save_path != "":
        fout = open(save_path, "w", encoding="utf8")
    with torch.no_grad():
        y_trues = []
        y_scores = []
        dres = dict()
        idx = 0
        df = pd.read_csv(testf)
        dcres, dqres = {"trues": [], "preds": []}, {"trues": [], "late_mean": [], "late_vote": [], "late_all": []}
        for i, row in df.iterrows():
            # print(f"idx: {idx}")
            # if idx == 2:
            #     import sys
            #     sys.exit()
            model.eval()

            dforget = dict() if model_name not in ["dkt_forget", "bakt_time"] else get_info_dkt_forget(row, data_config)

            concepts, responses = row["concepts"].split(","), row["responses"].split(",")
            ###
            # for AAAI competation
            rs = []
            for item in responses:
                newr = item if item != "-1" else "0" # default -1 to 0
                rs.append(newr)
            responses = rs
            ###
            curl = len(responses)

            # print("="*20)
            is_repeat = ["0"] * curl if "is_repeat" not in row else row["is_repeat"].split(",")
            is_repeat = [int(s) for s in is_repeat]
            questions = [] if "questions" not in row else row["questions"].split(",")
            times = [] if "timestamps" not in row else row["timestamps"].split(",")
           
            qlen, qtrainlen, ctrainlen = get_cur_teststart(is_repeat, train_ratio)
            cq = torch.tensor([int(s) for s in questions]).to(device)
            cc = torch.tensor([int(s) for s in concepts]).to(device)
            cr = torch.tensor([int(s) for s in responses]).to(device)
            ct = torch.tensor([int(s) for s in times]).to(device)
            dtotal = {"cq": cq, "cc": cc, "cr": cr, "ct": ct}
           
            curcin, currin = cc[0:ctrainlen].unsqueeze(0), cr[0:ctrainlen].unsqueeze(0)
            curqin = cq[0:ctrainlen].unsqueeze(0) if cq.shape[0] > 0 else cq
            curtin = ct[0:ctrainlen].unsqueeze(0) if ct.shape[0] > 0 else ct
           
            dcur = {"curqin": curqin, "curcin": curcin, "currin": currin, "curtin": curtin}
            
            curdforget = dict()
            for key in dforget:
                dforget[key] = torch.tensor(dforget[key]).to(device)
                curdforget[key] = dforget[key][0:ctrainlen].unsqueeze(0)
            # print(f"curcin: {curcin}")
            t = ctrainlen

            if not use_pred:
                uid, end = row["uid"], curl
                qidx = qtrainlen
                qidxs, ctrues, cpreds = predict_each_group2(dtotal, dcur, dforget, curdforget, is_repeat, qidx, uid, idx, model_name, model, t, end, fout, atkt_pad)
                # 计算
                save_currow_question_res(idx, dcres, dqres, qidxs, ctrues, cpreds, uid, fout)
            else:
                qidx = qtrainlen
                while t < curl:
                    rtmp = [t]
                    for k in range(t+1, curl):
                        if is_repeat[k] != 0:
                            rtmp.append(k)
                        else:
                            break

                    end = rtmp[-1]+1
                    uid = row["uid"]
                    
                    curqin, curcin, currin, curtin, curdforget, ctrues, cpreds = predict_each_group(dtotal, dcur, dforget, curdforget, is_repeat, qidx, uid, idx, model_name, model, t, end, fout, atkt_pad)
                    dcur = {"curqin": curqin, "curcin": curcin, "currin": currin, "curtin": curtin}
                    late_mean, late_vote, late_all = save_each_question_res(dcres, dqres, ctrues, cpreds)    
   
                    fout.write("\t".join([str(idx), str(uid), str(qidx), str(late_mean), str(late_vote), str(late_all)]) + "\n")      
                    t = end
                    qidx += 1
            idx += 1

        try: 
            dfinal = cal_predres(dcres, dqres)
            for key in dfinal:
                fout.write(key + "\t" + str(dfinal[key]) + "\n")
        except:
            print(f"can't output auc and accuracy!")
            dfinal = dict()
    return dfinal

def get_cur_teststart(is_repeat, train_ratio):
    curl = len(is_repeat)
    # print(is_repeat)
    qlen = is_repeat.count(0)
    qtrainlen = int(qlen * train_ratio)
    qtrainlen = 1 if qtrainlen == 0 else qtrainlen
    qtrainlen = qtrainlen - 1 if qtrainlen == qlen else qtrainlen
    # get real concept len
    ctrainlen, qidx = 0, 0
    i = 0
    while i < curl:
        if is_repeat[i] == 0:
            qidx += 1
        # print(f"i: {i}, curl: {curl}, qidx: {qidx}, qtrainlen: {qtrainlen}")
        # qtrainlen = 7 if qlen>7 else qtrainlen
        if qidx == qtrainlen:
            break
        i += 1
    for j in range(i+1, curl):
        if is_repeat[j] == 0:
            ctrainlen = j
            break
    return qlen, qtrainlen, ctrainlen

# def predict_each_group(curdforget, dforget, is_repeat, qidx, uid, idx, curqin, curcin, currin, model_name, model, t, cq, cc, cr, end, fout, atkt_pad=False, maxlen=200):
def predict_each_group(dtotal, dcur, dforget, curdforget, is_repeat, qidx, uid, idx, model_name, model, t, end, fout, atkt_pad=False, maxlen=200):
    """use the predict result as next question input
    """
    curqin, curcin, currin, curtin = dcur["curqin"], dcur["curcin"], dcur["currin"], dcur["curtin"]
    # print(f"cin8:{curcin}")
    # print(f"rin8:{currin}")
    cq, cc, cr, ct = dtotal["cq"], dtotal["cc"], dtotal["cr"], dtotal["ct"]
    

    nextcin, nextrin = curcin, currin
    import copy
    nextdforget = copy.deepcopy(curdforget)
    ctrues, cpreds = [], []
    for k in range(t, end):
        qin, cin, rin, tin = curqin, curcin, currin, curtin
        # print(f"cin9:{cin}")
        # print(f"rin9:{rin}")
        
        start = 0
        cinlen = cin.shape[1]
        if cinlen >= maxlen - 1:
            start = cinlen - maxlen + 1
        
        cin, rin = cin[:,start:], rin[:,start:]
        # print(f"cin10:{cin}")
        # print(f"rin10:{rin}")

        if cq.shape[0] > 0:
            qin = qin[:, start:]
        if ct.shape[0] > 0:
            tin = tin[:, start:]
        
        # print(f"start: {start}, cin: {cin.shape}")
        cout, true = cc.long()[k], cr.long()[k] 
        qout = None if cq.shape[0] == 0 else cq.long()[k]
        tout = None if ct.shape[0] == 0 else ct.long()[k]
        
        if qout != None:
            curq = torch.tensor([[qout.item()]]).to(device)
            qin = torch.cat((qin, curq), axis=1)
        curc, curr = torch.tensor([[cout.item()]]).to(device), torch.tensor([[1]]).to(device)
        cin, rin = torch.cat((cin, curc), axis=1), torch.cat((rin, curr), axis=1)

        y, reg_loss = model(cin.long(), rin.long(), qin.long())
        pred = y[0][-1]
        
        
        predl = 1 if pred.item() >= 0.5 else 0
        cpred = torch.tensor([[predl]]).to(device)

        nextqin = cq[0:k+1].unsqueeze(0) if cq.shape[0] > 0 else qin
        nexttin = ct[0:k+1].unsqueeze(0) if ct.shape[0] > 0 else tin
        nextcin = cc[0:k+1].unsqueeze(0)
        nextrin = torch.cat((nextrin, cpred), axis=1)### change!!
        
        # save pred res
        ctrues.append(true.item())
        cpreds.append(pred.item())

        # output
        clist, rlist = cin.squeeze(0).long().tolist()[0:k], rin.squeeze(0).long().tolist()[0:k]
        # print("\t".join([str(idx), str(uid), str(k), str(qidx), str(is_repeat[t:end]), str(len(clist)), str(clist), str(rlist), str(cout.item()), str(true.item()), str(pred.item()), str(predl)]))
        fout.write("\t".join([str(idx), str(uid), str(k), str(qidx), str(is_repeat[t:end]), str(len(clist)), str(clist), str(rlist), str(cout.item()), str(true.item()), str(pred.item()), str(predl)]) + "\n")
    # nextcin, nextrin = nextcin.unsqueeze(0), nextrin.unsqueeze(0)
    return nextqin, nextcin, nextrin, nexttin, nextdforget, ctrues, cpreds

def save_each_question_res(dcres, dqres, ctrues, cpreds):
    # save res
    high, low = [], []
    for true, pred in zip(ctrues, cpreds):
        dcres["trues"].append(true)
        dcres["preds"].append(pred)
        if pred >= 0.5:
            high.append(pred)
        else:
            low.append(pred)
    cpreds = np.array(cpreds)
    late_mean = np.mean(cpreds)
    correctnum = list(cpreds>=0.5).count(True)
    late_vote = np.mean(high) if correctnum / len(cpreds) >= 0.5 else np.mean(low)
    late_all = np.mean(high) if correctnum == len(cpreds) else np.mean(low)
    assert len(set(ctrues)) == 1
    dqres["trues"].append(dcres["trues"][-1])
    dqres["late_mean"].append(late_mean)
    dqres["late_vote"].append(late_vote)
    dqres["late_all"].append(late_all)
    return late_mean, late_vote, late_all

def cal_predres(dcres, dqres):
    dres = dict()#{"concept": [], "late_mean": [], "late_vote": [], "late_all": []}

    ctrues, cpreds = np.array(dcres["trues"]), np.array(dcres["preds"])
    # print(f"key: concepts, ts.shape: {ctrues.shape}, ps.shape: {cpreds.shape}")
    auc = metrics.roc_auc_score(y_true=ctrues, y_score=cpreds)
    prelabels = [1 if p >= 0.5 else 0 for p in cpreds]
    acc = metrics.accuracy_score(ctrues, prelabels)

    dres["concepts"] = [len(cpreds), auc, acc]

    qtrues = np.array(dqres["trues"])
    for key in dqres:
        if key == "trues":
            continue
        preds = np.array(dqres[key])
        # print(f"key: {key}, ts.shape: {qtrues.shape}, ps.shape: {preds.shape}")
        auc = metrics.roc_auc_score(y_true=qtrues, y_score=preds)
        prelabels = [1 if p >= 0.5 else 0 for p in preds]
        acc = metrics.accuracy_score(qtrues, prelabels)
        dres[key] = [len(preds), auc, acc]
    return dres

def prepare_data(model_name, is_repeat, qidx, dcur, curdforget, dtotal, dforget, t, end, maxlen=200):
    curqin, curcin, currin, curtin = dcur["curqin"], dcur["curcin"], dcur["currin"], dcur["curtin"]
    cq, cc, cr, ct = dtotal["cq"], dtotal["cc"], dtotal["cr"], dtotal["ct"]
    dqshfts, dcshfts, drshfts, dtshfts, dds, ddshfts = [], [], [], [], dict(), dict()
    dqs, dcs, drs, dts = [], [], [], []
    
    qidxs = []
    qstart = qidx-1
    for k in range(t, end):
        if is_repeat[k] == 0:
            qstart += 1
            qidxs.append(qstart)
        else:
            qidxs.append(qstart)
        # get start
        start = 0
        cinlen = curcin.shape[1]
        if cinlen >= maxlen - 1:
            start = cinlen - maxlen + 1

        curc, curr = cc.long()[k], cr.long()[k]
        curc, curr = torch.tensor([[curc.item()]]).to(device), torch.tensor([[curr.item()]]).to(device)
        dcs.append(curcin[:, start:])
        drs.append(currin[:, start:])

        curc, curr = torch.cat((curcin[:, start+1:], curc), axis=1), torch.cat((currin[:, start+1:], curr), axis=1)
        dcshfts.append(curc)
        drshfts.append(curr)
        if cq.shape[0] > 0:
            curq = cq.long()[k]
            curq = torch.tensor([[curq.item()]]).to(device)

            dqs.append(curqin[:, start:])
            curq = torch.cat((curqin[:, start+1:], curq), axis=1)
            dqshfts.append(curq)
        if ct.shape[0] > 0:
            curt = ct.long()[k]
            curt = torch.tensor([[curt.item()]]).to(device)

            dts.append(curtin[:, start:])
            curt = torch.cat((curtin[:, start+1:], curt), axis=1)
            dtshfts.append(curt)
          

        d, dshft = dict(), dict()
        
        
    finalcs, finalrs = torch.cat(dcs, axis=0), torch.cat(drs, axis=0)
    finalqs, finalqshfts = torch.tensor([]), torch.tensor([])
    finalts, finaltshfts = torch.tensor([]), torch.tensor([]) 
    if cq.shape[0] > 0:
        finalqs = torch.cat(dqs, axis=0)
        finalqshfts = torch.cat(dqshfts, axis=0)
    if ct.shape[0] > 0:
        finalts = torch.cat(dts, axis=0)
        finaltshfts = torch.cat(dtshfts, axis=0)
    finalcshfts, finalrshfts = torch.cat(dcshfts, axis=0), torch.cat(drshfts, axis=0)
    finald, finaldshft = dict(), dict()
    for key in dds:
        finald[key] = torch.cat(dds[key], axis=0)
        finaldshft[key] = torch.cat(ddshfts[key], axis=0)
    # print(f"qidx: {len(qidxs)}, finalqs: {finalqs.shape}, finalcs: {finalcs.shape}, finalrs: {finalrs.shape}")
    # print(f"qidx: {len(qidxs)}, finalqshfts: {finalqshfts.shape}, finalcshfts: {finalcshfts.shape}, finalrshfts: {finalrshfts.shape}")


    return qidxs, finalqs, finalcs, finalrs, finalts, finalqshfts, finalcshfts, finalrshfts, finaltshfts, finald, finaldshft

# def predict_each_group2(curdforget, dforget, is_repeat, qidx, uid, idx, curqin, curcin, currin, model_name, model, t, cq, cc, cr, end, fout, atkt_pad=False, maxlen=200):
def predict_each_group2(dtotal, dcur, dforget, curdforget, is_repeat, qidx, uid, idx, model_name, model, t, end, fout, atkt_pad=False, maxlen=200):
    """not use the predict result
    """
    curqin, curcin, currin, curtin = dcur["curqin"], dcur["curcin"], dcur["currin"], dcur["curtin"]
    cq, cc, cr, ct = dtotal["cq"], dtotal["cc"], dtotal["cr"], dtotal["ct"]
   
    nextcin, nextrin = curcin, currin
    import copy
    nextdforget = copy.deepcopy(curdforget)
    ctrues, cpreds = [], []
    qidxs, finalqs, finalcs, finalrs, finalts, finalqshfts, finalcshfts, finalrshfts, finaltshfts, finald, finaldshft = prepare_data(model_name, is_repeat, qidx, dcur, curdforget, dtotal, dforget, t, end)
    bidx, bz = 0, 128
    while bidx < finalcs.shape[0]:
        curc, curr = finalcs[bidx: bidx+bz], finalrs[bidx: bidx+bz]
        curcshft, currshft = finalcshfts[bidx: bidx+bz], finalrshfts[bidx: bidx+bz]
        curqidxs = qidxs[bidx: bidx+bz]
        curq, curqshft = torch.tensor([[]]), torch.tensor([[]])
        if finalqs.shape[0] > 0:
            curq = finalqs[bidx: bidx+bz]
            curqshft = finalqshfts[bidx: bidx+bz]
        curt, curtshft = torch.tensor([[]]), torch.tensor([[]])
        if finalts.shape[0] > 0:
            curt = finalts[bidx: bidx+bz]
            curtshft = finaltshfts[bidx: bidx+bz]
        curd, curdshft = dict(), dict()
                 

        ccq = torch.cat((curq[:,0:1], curqshft), dim=1)
        ccc = torch.cat((curc[:,0:1], curcshft), dim=1)
        ccr = torch.cat((curr[:,0:1], currshft), dim=1)
        cct = torch.cat((curt[:,0:1], curtshft), dim=1)
   
        
        y, reg_loss = model(ccc.long(), ccr.long(), ccq.long())
        y = y[:,1:]
        
        pred = y[:, -1].tolist()
        true = ccr[:, -1].tolist()
    
        # save pred res
        ctrues.extend(true)
        cpreds.extend(pred)

        # output
        
        for i in range(0, curc.shape[0]):
            clist, rlist = curc[i].long().tolist()[0:t], curr[i].long().tolist()[0:t]
            cshftlist, rshftlist = curcshft[i].long().tolist()[0:t], currshft[i].long().tolist()[0:t]
            qidx = curqidxs[i]
            predl = 1 if pred[i] >= 0.5 else 0
            fout.write("\t".join([str(idx), str(uid), str(bidx+i), str(qidx), str(len(clist)), str(clist), str(rlist), str(cshftlist), str(rshftlist), str(true[i]), str(pred[i]), str(predl)]) + "\n")

        bidx += bz
    return qidxs, ctrues, cpreds

def save_currow_question_res(idx, dcres, dqres, qidxs, ctrues, cpreds, uid, fout):
    # save res
    dqidx = dict()
    # dhigh, dlow = dict(), dict()
    for i in range(0, len(qidxs)):
        true, pred = ctrues[i], cpreds[i]
        qidx = qidxs[i]
        dqidx.setdefault(qidx, {"trues": [], "preds": []})
        dqidx[qidx]["trues"].append(true)
        dqidx[qidx]["preds"].append(pred)

    for qidx in dqidx:
        ctrues, cpreds = dqidx[qidx]["trues"], dqidx[qidx]["preds"]
        late_mean, late_vote, late_all = save_each_question_res(dcres, dqres, ctrues, cpreds)
        fout.write("\t".join([str(idx), str(uid), str(qidx), str(late_mean), str(late_vote), str(late_all)]) + "\n")

