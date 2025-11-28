import os
import sys
import argparse
import pandas as pd
import csv
import numpy as np
import json
import copy
from sklearn.cluster import KMeans

# ==============================================================================
# SECTION 0: UTILITY FUNCTIONS (Reconstructed from .utils)
# ==============================================================================

def format_list2str(input_list):
    """Formats a list of elements into a comma-separated string."""
    return ",".join([str(x) for x in input_list])

def write_txt(file, data):
    """
    Writes user interaction data to a text file in the standard 6-line format.
    Format:
    1. uid, sequence_length
    2. problem_ids
    3. skill_ids
    4. responses (correctness)
    5. types (answer types)
    6. usetimes (response costs/timestamps - NOW DISCRETIZED)
    """
    with open(file, 'w') as f:
        for item in data:
            # item[0] is [uid, len]
            f.write(",".join(item[0]) + "\n")
            # item[1] is problems
            f.write(item[1] + "\n")
            # item[2] is skills
            f.write(item[2] + "\n")
            # item[3] is ans
            f.write(item[3] + "\n")
            # item[4] is type list
            f.write(",".join([str(x) for x in item[4]]) if isinstance(item[4], list) else item[4])
            f.write("\n")
            # item[5] is usetimes list
            f.write(",".join([str(x) for x in item[5]]) if isinstance(item[5], list) else item[5])
            f.write("\n")

def sta_infos(df, keys, stares):
    """
    Calculates and prints statistics about the dataframe.
    keys: list of column names [user_id, concept_id, problem_id]
    """
    # Determine keys based on input length
    uid = keys[0]
    cid = keys[1]
    qid = keys[2] if len(keys) > 2 else None

    ins_num = len(df)
    us_num = df[uid].nunique()
    cs_num = df[cid].nunique() if cid in df else 0
    qs_num = df[qid].nunique() if qid and qid in df else 0

    avgins = ins_num / us_num if us_num > 0 else 0
    
    if qid and cid in df:
        # Average concepts per question
        avgcq = df.groupby(qid)[cid].nunique().mean()
    else:
        avgcq = 0

    na_num = df.isnull().sum().sum()

    stares.append(f"interaction num: {ins_num}, user num: {us_num}, question num: {qs_num}, concept num: {cs_num}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na_num}")
    
    return ins_num, us_num, qs_num, cs_num, avgins, avgcq, na_num

def discretize_continuos_values(df, col_name, n_clusters=20):
    """
    Apply Log transformation and K-Means clustering to discretize a continuous column.
    Same logic as compute_assist2009_avg_rt.
    """
    print(f"   [Process] Discretizing column '{col_name}' using Log + KMeans(k={n_clusters})...")
    
    # 1. Filter valid positive values for Log transform
    # Convert to numeric, errors to NaN
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    
    # Create a mask for valid processing
    mask = (df[col_name].notna()) & (df[col_name] > 0)
    
    if mask.sum() < n_clusters:
        print(f"   [Warning] Not enough valid data points ({mask.sum()}) for {n_clusters} clusters. Using raw or simple binning.")
        df['discrete_time'] = df[col_name].fillna(0).astype(int)
        return df

    raw_values = df.loc[mask, col_name].values
    
    # 2. Log Transformation
    log_values = np.log(raw_values)
    
    # 3. K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(log_values.reshape(-1, 1))
    
    # 4. Sort clusters so that larger ID means larger time
    cluster_centers = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(cluster_centers)
    label_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
    
    # 5. Apply back to dataframe
    original_labels = kmeans.predict(log_values.reshape(-1, 1))
    final_labels = [label_map[l] for l in original_labels]
    
    # Initialize discrete column with a default value (e.g., 0 or -1 for invalid times)
    # Here we use 0 for invalid/zero times, and map clusters to 0-(k-1) or 1-k?
    # compute_assist2009_avg_rt produced 0-19. We will stick to that.
    # Note: If time is missing or <=0, we map it to the smallest cluster (0) or a specific token.
    # Let's assign 0 to invalid times.
    df['discrete_time'] = 0 
    df.loc[mask, 'discrete_time'] = final_labels
    
    print("   [Success] Discretization complete.")
    return df

# ==============================================================================
# SECTION 1: DATASET SPECIFIC PREPROCESSORS
# ==============================================================================

# --- ASSIST2009 Logic ---
def process_assist2009(read_file, write_file):
    KEYS = ["user_id", "skill_id", "problem_id"]
    stares = []

    print(f"Reading Assist2009 from {read_file}...")
    df = pd.read_csv(read_file, encoding = 'utf-8', dtype=str)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    df['tmp_index'] = range(len(df))
    _df = df.dropna(subset=["user_id","problem_id", "skill_id", "correct", "order_id"])

    # --- NEW: Discretize usetimes (ms_first_response) ---
    # Apply discretization globally before grouping
    if 'ms_first_response' in _df.columns:
        _df = discretize_continuos_values(_df, 'ms_first_response', n_clusters=20)
    else:
        print("[Warning] ms_first_response column not found, using 0 for usetimes.")
        _df['discrete_time'] = 0
    # ----------------------------------------------------

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(_df, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    ui_df = _df.groupby('user_id', sort=False)

    user_inters = []
    for ui in ui_df:
        user, tmp_inter = ui[0], ui[1]
        tmp_inter = tmp_inter.sort_values(by=['order_id','tmp_index'])
        seq_len = len(tmp_inter)
        seq_problems = tmp_inter['problem_id'].tolist()
        seq_skills = tmp_inter['skill_id'].tolist()
        seq_ans = tmp_inter['correct'].tolist()
        seq_type = tmp_inter['answer_type'].tolist()
        
        # --- MODIFIED: Use the discretized column ---
        # seq_usetimes = tmp_inter['ms_first_response'].tolist()
        # seq_usetimes = [int(x) for x in seq_usetimes] # Old logic
        seq_usetimes = tmp_inter['discrete_time'].tolist() # New logic: KMeans Labels
        
        assert seq_len == len(seq_problems) == len(seq_skills) == len(seq_ans)

        user_inters.append(
            [[str(user), str(seq_len)], format_list2str(seq_problems), format_list2str(seq_skills), format_list2str(seq_ans), seq_type, seq_usetimes])

    write_txt(write_file, user_inters)
    print("\n".join(stares))

# --- ASSIST2015/2017 Logic ---
def process_assist2017(read_file, write_file):
    df = pd.read_csv(read_file, encoding='utf-8', dtype=str)
    keys = ["studentId", "skill", "problemId"]
    stares = []
    
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, keys, stares)
    print(
        f"original interaction num: {df.shape[0]}, user num: {df['studentId'].nunique()}, question num: {df['problemId'].nunique()}, "
        f"concept num: {df['skill'].nunique()}, avg(ins) per s:{avgins}, avg(c) per q:{avgcq}, na:{na}")

    df["index"] = range(len(df))

    df = df.dropna(subset=["studentId", "problemId", "correct", "skill", "startTime"])
    df = df[df['correct'].isin(['0', '1'])] # check string '0','1' if dtype=str
    
    # Handle timeTaken conversion for discretization
    # Note: Original logic did: round(float(x) * 1000)
    # We will compute raw values first for the discretizer
    df.loc[:, 'raw_timeTaken'] = df['timeTaken'].apply(lambda x: float(x) * 1000 if x and x != 'NA' else 0)

    # --- NEW: Discretize usetimes (timeTaken) ---
    df = discretize_continuos_values(df, 'raw_timeTaken', n_clusters=20)
    # --------------------------------------------

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, keys, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    df2 = df[["index", "studentId", "problemId", "skill", "correct", "discrete_time", "startTime", "problemType"]]
    ui_df = df2.groupby('studentId', sort=False)

    user_inter = []
    for ui in ui_df:
        user, tmp_inter = ui[0], ui[1]  
        tmp_inter.loc[:, 'startTime'] = tmp_inter.loc[:, 'startTime'].apply(lambda t: int(float(t)) * 1000)
        tmp_inter = tmp_inter.sort_values(by=['startTime', 'index'])

        seq_len = len(tmp_inter)
        seq_problems = tmp_inter['problemId'].tolist()
        seq_skills = tmp_inter['skill'].tolist()
        seq_ans = tmp_inter['correct'].tolist()
        seq_type = tmp_inter['problemType'].tolist()
        
        # --- MODIFIED: Use discretized column ---
        seq_usetimes = tmp_inter['discrete_time'].tolist()

        assert seq_len == len(seq_problems) == len(seq_skills) == len(seq_ans) == len(seq_type) == len(seq_usetimes)

        # In this context: item[4]=submit_time (as type/timestamp), item[5]=response_cost
        user_inter.append(
            [[str(user), str(seq_len)], format_list2str(seq_problems), format_list2str(seq_skills), format_list2str(seq_ans), seq_type, seq_usetimes])

    write_txt(write_file, user_inter)
    print("\n".join(stares))


def process_aaai2023(read_file, write_file, dq2t):
    KEYS = ["uid", "concept_id", "question_id"]

    stares = []
    df = pd.read_csv(read_file, low_memory=False)
    # 合并知识点信息
    ts = []
    for i, row in df.iterrows():
        qid = row["question_id"]
        tid = dq2t[qid]
        ts.append(tid)
    df["type"] = ts

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    df = discretize_continuos_values(df, 'response_time', n_clusters=20)


    ui_df = df.groupby('uid', sort=False)

    user_inters = []
    for ui in ui_df:
        user, tmp_inter = ui[0], ui[1]
        tmp_inter = tmp_inter.sort_values(by=["timestamp"])
        seq_len = len(tmp_inter)
        seq_skills = tmp_inter['concept_id'].tolist()
        seq_ans = tmp_inter['responses'].tolist()
        seq_problems = tmp_inter['question_id'].tolist()
        seq_type = tmp_inter['type'].tolist()
        
        # --- MODIFIED: Use discretized column ---
        seq_usetimes = tmp_inter['discrete_time'].tolist()

        assert seq_len == len(seq_skills) == len(seq_ans)


        user_inters.append(
            [[str(user), str(seq_len)], format_list2str(seq_problems), format_list2str(seq_skills), format_list2str(seq_ans), seq_type, seq_usetimes])


    write_txt(write_file, user_inters)

    print("\n".join(stares))

    return

def load_q2c(fname):
    dq2t = dict()
    with open("../autodl-tmp/data/aaai2023/keyid2idx.json", "r") as fin:
        obj = json.load(fin)
        keyid2idx = obj["questions"]
    with open(fname, "r") as fin:
        obj = json.load(fin)
        for qid in obj:
            cur = obj[qid]
            types = cur["type"]
            dq2t[keyid2idx[qid]] = types
    return dq2t


# ==============================================================================
# SECTION 2: GLOBAL CONSTANTS & BASE UTILITIES (Originally from split_datasets.py)
# ==============================================================================

ALL_KEYS = ["fold", "uid", "questions", "concepts", "responses", "type",
            "usetimes", "selectmasks", "is_repeat", "qidxs", "rest", "orirow", "cidxs"]
ONE_KEYS = ["fold", "uid"]

def read_data(fname, min_seq_len=3, response_set=[0, 1]):
    effective_keys = set()
    dres = dict()
    delstu, delnum, badr = 0, 0, 0
    goodnum = 0
    with open(fname, "r", encoding="utf8") as fin:
        i = 0
        lines = fin.readlines()
        dcur = dict()
        while i < len(lines):
            line = lines[i].strip()
            if i % 6 == 0:  # stuid
                effective_keys.add("uid")
                tmps = line.split(",")
                if "(" in tmps[0]:
                    stuid, seq_len = tmps[0].replace('(', ''), int(tmps[2])
                else:
                    stuid, seq_len = tmps[0], int(tmps[1])
                if seq_len < min_seq_len:  # delete use seq len less than min_seq_len
                    i += 6
                    dcur = dict()
                    delstu += 1
                    delnum += seq_len
                    continue
                dcur["uid"] = stuid
                goodnum += seq_len
            elif i % 6 == 1:  # question ids / names
                qs = []
                if line.find("NA") == -1:
                    effective_keys.add("questions")
                    qs = line.split(",")
                dcur["questions"] = qs
            elif i % 6 == 2:  # concept ids / names
                cs = []
                if line.find("NA") == -1:
                    effective_keys.add("concepts")
                    cs = line.split(",")
                dcur["concepts"] = cs
            elif i % 6 == 3:  # responses
                effective_keys.add("responses")
                rs = []
                if line.find("NA") == -1:
                    flag = True
                    for r in line.split(","):
                        try:
                            r = int(r)
                            if r not in response_set:  # check if r in response set.
                                print(f"error response in line: {i}")
                                flag = False
                                break
                            rs.append(r)
                        except:
                            print(f"error response in line: {i}")
                            flag = False
                            break
                    if not flag:
                        i += 3
                        dcur = dict()
                        badr += 1
                        continue
                dcur["responses"] = rs
            elif i % 6 == 4:  # type
                ts = []
                if line.find("NA") == -1:
                    effective_keys.add("type")
                    ts = line.split(",")
                dcur["type"] = ts
            elif i % 6 == 5:  # times (usetimes)
                usets = []
                if line.find("NA") == -1:
                    effective_keys.add("usetimes")
                    usets = line.split(",")
                dcur["usetimes"] = usets

                for key in effective_keys:
                    dres.setdefault(key, [])
                    if key != "uid":
                        dres[key].append(",".join([str(k) for k in dcur[key]]))
                    else:
                        dres[key].append(dcur[key])
                dcur = dict()
            i += 1
    df = pd.DataFrame(dres)
    print(
        f"delete bad stu num of len: {delstu}, delete interactions: {delnum}, of r: {badr}, good num: {goodnum}")
    return df, effective_keys

def extend_multi_concepts(df, effective_keys):
    if "questions" not in effective_keys or "concepts" not in effective_keys:
        print("has no questions or concepts! return original.")
        return df, effective_keys
    extend_keys = set(df.columns) - {"uid"}

    dres = {"uid": df["uid"]}
    for _, row in df.iterrows():
        dextend_infos = dict()
        for key in extend_keys:
            dextend_infos[key] = row[key].split(",")
        dextend_res = dict()
        for i in range(len(dextend_infos["questions"])):
            dextend_res.setdefault("is_repeat", [])
            if dextend_infos["concepts"][i].find("_") != -1:
                ids = dextend_infos["concepts"][i].split("_")
                dextend_res.setdefault("concepts", [])
                dextend_res["concepts"].extend(ids)
                for key in extend_keys:
                    if key != "concepts":
                        dextend_res.setdefault(key, [])
                        dextend_res[key].extend(
                            [dextend_infos[key][i]] * len(ids))
                dextend_res["is_repeat"].extend(
                    ["0"] + ["1"] * (len(ids) - 1))  # 1: repeat, 0: original
            else:
                for key in extend_keys:
                    dextend_res.setdefault(key, [])
                    dextend_res[key].append(dextend_infos[key][i])
                dextend_res["is_repeat"].append("0")
        for key in dextend_res:
            dres.setdefault(key, [])
            dres[key].append(",".join(dextend_res[key]))

    finaldf = pd.DataFrame(dres)
    effective_keys.add("is_repeat")
    return finaldf, effective_keys

def id_mapping(df):
    """
    Maps string IDs to integer indices.
    Modified: Now includes 'type' in mapping if it exists.
    """
    id_keys = ["questions", "concepts", "uid", "type"] # Added type
    dres = dict()
    dkeyid2idx = dict()
    print(f"df.columns: {df.columns}")
    for key in df.columns:
        if key not in id_keys:
            # This handles 'usetimes' automatically (keeps them as is)
            dres[key] = df[key]
    for i, row in df.iterrows():
        for key in id_keys:
            if key not in df.columns:
                continue
            dkeyid2idx.setdefault(key, dict())
            dres.setdefault(key, [])
            curids = []
            # Check if value is a string before splitting, handle potential NA/float issues
            val_str = str(row[key]) if pd.notna(row[key]) else ""
            for id in val_str.split(","):
                if id == "": continue 
                if id not in dkeyid2idx[key]:
                    dkeyid2idx[key][id] = len(dkeyid2idx[key])
                curids.append(str(dkeyid2idx[key][id]))
            dres[key].append(",".join(curids))
    finaldf = pd.DataFrame(dres)
    return finaldf, dkeyid2idx

def train_test_split(df, test_ratio=0.2):
    df = df.sample(frac=1.0, random_state=1024)
    datanum = df.shape[0]
    test_num = int(datanum * test_ratio)
    train_num = datanum - test_num
    train_df = df[0:train_num]
    test_df = df[train_num:]
    print(
        f"total num: {datanum}, train+valid num: {train_num}, test num: {test_num}")
    return train_df, test_df

def KFold_split(df, k=5):
    df = df.sample(frac=1.0, random_state=1024)
    datanum = df.shape[0]
    test_ratio = 1 / k
    test_num = int(datanum * test_ratio)
    rest = datanum % k

    start = 0
    folds = []
    for i in range(0, k):
        if rest > 0:
            end = start + test_num + 1
            rest -= 1
        else:
            end = start + test_num
        folds.extend([i] * (end - start))
        print(f"fold: {i+1}, start: {start}, end: {end}, total num: {datanum}")
        start = end
    finaldf = copy.deepcopy(df)
    finaldf["fold"] = folds
    return finaldf

def save_dcur(row, effective_keys):
    dcur = dict()
    for key in effective_keys:
        if key not in ONE_KEYS:
            dcur[key] = row[key].split(",")
        else:
            dcur[key] = row[key]
    return dcur

def generate_sequences(df, effective_keys, min_seq_len=3, maxlen=200, pad_val=-1):
    save_keys = list(effective_keys) + ["selectmasks"]
    dres = {"selectmasks": []}
    dropnum = 0
    for i, row in df.iterrows():
        dcur = save_dcur(row, effective_keys)

        rest, lenrs = len(dcur["responses"]), len(dcur["responses"])
        j = 0
        while lenrs >= j + maxlen:
            rest = rest - (maxlen)
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    dres[key].append(",".join(dcur[key][j: j + maxlen]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(",".join(["1"] * maxlen))

            j += maxlen
        if rest < min_seq_len:  # delete sequence len less than min_seq_len
            dropnum += rest
            continue

        pad_dim = maxlen - rest
        for key in effective_keys:
            dres.setdefault(key, [])
            if key not in ONE_KEYS:
                paded_info = np.concatenate(
                    [dcur[key][j:], np.array([pad_val] * pad_dim)])
                dres[key].append(",".join([str(k) for k in paded_info]))
            else:
                dres[key].append(dcur[key])
        dres["selectmasks"].append(
            ",".join(["1"] * rest + [str(pad_val)] * pad_dim))

    dfinal = dict()
    for key in ALL_KEYS:
        if key in save_keys:
            dfinal[key] = dres[key]
    finaldf = pd.DataFrame(dfinal)
    print(f"dropnum: {dropnum}")
    return finaldf

def generate_window_sequences(df, effective_keys, maxlen=200, pad_val=-1):
    save_keys = list(effective_keys) + ["selectmasks"]
    dres = {"selectmasks": []}
    for i, row in df.iterrows():
        dcur = save_dcur(row, effective_keys)
        lenrs = len(dcur["responses"])
        if lenrs > maxlen:
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    dres[key].append(",".join(dcur[key][0: maxlen]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(",".join(["1"] * maxlen))
            for j in range(maxlen+1, lenrs+1):
                for key in effective_keys:
                    dres.setdefault(key, [])
                    if key not in ONE_KEYS:
                        dres[key].append(",".join([str(k)
                                         for k in dcur[key][j-maxlen: j]]))
                    else:
                        dres[key].append(dcur[key])
                dres["selectmasks"].append(
                    ",".join([str(pad_val)] * (maxlen - 1) + ["1"]))
        else:
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    pad_dim = maxlen - lenrs
                    paded_info = np.concatenate(
                        [dcur[key][0:], np.array([pad_val] * pad_dim)])
                    dres[key].append(",".join([str(k) for k in paded_info]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(
                ",".join(["1"] * lenrs + [str(pad_val)] * pad_dim))

    dfinal = dict()
    for key in ALL_KEYS:
        if key in save_keys:
            dfinal[key] = dres[key]
    finaldf = pd.DataFrame(dfinal)
    return finaldf

def get_inter_qidx(df):
    """add global id for each interaction"""
    qidx_ids = []
    bias = 0
    inter_num = 0
    for _, row in df.iterrows():
        ids_list = [str(x+bias)
                    for x in range(len(row['responses'].split(',')))]
        inter_num += len(ids_list)
        ids = ",".join(ids_list)
        qidx_ids.append(ids)
        bias += len(ids_list)
    assert inter_num-1 == int(ids_list[-1])
    return qidx_ids

def add_qidx(dcur, global_qidx):
    idxs, rests = [], []
    for r in dcur["is_repeat"]:
        if str(r) == "0":
            global_qidx += 1
        idxs.append(global_qidx)
    for i in range(0, len(idxs)):
        rests.append(idxs[i+1:].count(idxs[i]))
    return idxs, rests, global_qidx

def expand_question(dcur, global_qidx, pad_val=-1):
    dextend, dlast = dict(), dict()
    repeats = dcur["is_repeat"]
    last = -1
    dcur["qidxs"], dcur["rest"], global_qidx = add_qidx(dcur, global_qidx)
    for i in range(len(repeats)):
        if str(repeats[i]) == "0":
            for key in dcur.keys():
                if key in ONE_KEYS:
                    continue
                dlast[key] = dcur[key][0: i]
        if i == 0:
            for key in dcur.keys():
                if key in ONE_KEYS:
                    continue
                dextend.setdefault(key, [])
                dextend[key].append([dcur[key][0]])
            dextend.setdefault("selectmasks", [])
            dextend["selectmasks"].append([pad_val])
        else:
            for key in dcur.keys():
                if key in ONE_KEYS:
                    continue
                dextend.setdefault(key, [])
                if last == "0" and str(repeats[i]) == "0":
                    dextend[key][-1] += [dcur[key][i]]
                else:
                    dextend[key].append(dlast[key] + [dcur[key][i]])
            dextend.setdefault("selectmasks", [])
            if last == "0" and str(repeats[i]) == "0":
                dextend["selectmasks"][-1] += [1]
            elif len(dlast["responses"]) == 0:  # the first question
                dextend["selectmasks"].append([pad_val])
            else:
                dextend["selectmasks"].append(
                    len(dlast["responses"]) * [pad_val] + [1])
        last = str(repeats[i])
    return dextend, global_qidx

def generate_question_sequences(df, effective_keys, window=True, min_seq_len=3, maxlen=200, pad_val=-1):
    if "questions" not in effective_keys or "concepts" not in effective_keys:
        print(f"has no questions or concepts, has no question sequences!")
        return False, None
    save_keys = list(effective_keys) + \
        ["selectmasks", "qidxs", "rest", "orirow"]
    dres = {}
    global_qidx = -1
    df["index"] = list(range(0, df.shape[0]))
    for i, row in df.iterrows():
        dcur = save_dcur(row, effective_keys)
        dcur["orirow"] = [row["index"]] * len(dcur["responses"])

        dexpand, global_qidx = expand_question(dcur, global_qidx)
        seq_num = len(dexpand["responses"])
        for j in range(seq_num):
            curlen = len(dexpand["responses"][j])
            if curlen < 2:  # 不预测第一个题
                continue
            if curlen < maxlen:
                for key in dexpand:
                    pad_dim = maxlen - curlen
                    paded_info = np.concatenate(
                        [dexpand[key][j][0:], np.array([pad_val] * pad_dim)])
                    dres.setdefault(key, [])
                    dres[key].append(",".join([str(k) for k in paded_info]))
                for key in ONE_KEYS:
                    dres.setdefault(key, [])
                    dres[key].append(dcur[key])
            else:
                # window
                if window:
                    if dexpand["selectmasks"][j][maxlen-1] == 1:
                        for key in dexpand:
                            dres.setdefault(key, [])
                            dres[key].append(
                                ",".join([str(k) for k in dexpand[key][j][0:maxlen]]))
                        for key in ONE_KEYS:
                            dres.setdefault(key, [])
                            dres[key].append(dcur[key])

                    for n in range(maxlen+1, curlen+1):
                        if dexpand["selectmasks"][j][n-1] == 1:
                            for key in dexpand:
                                dres.setdefault(key, [])
                                if key == "selectmasks":
                                    dres[key].append(
                                        ",".join([str(pad_val)] * (maxlen - 1) + ["1"]))
                                else:
                                    dres[key].append(
                                        ",".join([str(k) for k in dexpand[key][j][n-maxlen: n]]))
                            for key in ONE_KEYS:
                                dres.setdefault(key, [])
                                dres[key].append(dcur[key])
                else:
                    # not window
                    k = 0
                    rest = curlen
                    while curlen >= k + maxlen:
                        rest = rest - maxlen
                        if dexpand["selectmasks"][j][k + maxlen - 1] == 1:
                            for key in dexpand:
                                dres.setdefault(key, [])
                                dres[key].append(
                                    ",".join([str(s) for s in dexpand[key][j][k: k + maxlen]]))
                            for key in ONE_KEYS:
                                dres.setdefault(key, [])
                                dres[key].append(dcur[key])
                        k += maxlen
                    if rest < min_seq_len:  # 剩下长度<min_seq_len不预测
                        continue
                    pad_dim = maxlen - rest
                    for key in dexpand:
                        dres.setdefault(key, [])
                        paded_info = np.concatenate(
                            [dexpand[key][j][k:], np.array([pad_val] * pad_dim)])
                        dres[key].append(",".join([str(s)
                                         for s in paded_info]))
                    for key in ONE_KEYS:
                        dres.setdefault(key, [])
                        dres[key].append(dcur[key])

    dfinal = dict()
    for key in ALL_KEYS:
        if key in save_keys:
            dfinal[key] = dres[key]
    finaldf = pd.DataFrame(dfinal)
    return True, finaldf

def save_id2idx(dkeyid2idx, save_path):
    with open(save_path, "w+") as fout:
        fout.write(json.dumps(dkeyid2idx))

def write_config(dataset_name, dkeyid2idx, effective_keys, configf, dpath, k=5, min_seq_len=3, maxlen=200, flag=False, other_config={}):
    """
    Writes the dataset configuration.
    Modified: Includes 'num_type' count if 'type' is present.
    """
    input_type, num_q, num_c, num_type = [], 0, 0, 0
    if "questions" in effective_keys:
        input_type.append("questions")
        num_q = len(dkeyid2idx["questions"])
    if "concepts" in effective_keys:
        input_type.append("concepts")
        num_c = len(dkeyid2idx["concepts"])
    if "type" in effective_keys:
        input_type.append("type")
        num_type = len(dkeyid2idx["type"])

    folds = list(range(0, k))
    dconfig = {
        "dpath": dpath,
        "num_q": num_q,
        "num_c": num_c,
        "num_type": num_type, # Added num_type
        "input_type": input_type,
        "max_concepts": dkeyid2idx.get("max_concepts", 1),
        "min_seq_len": min_seq_len,
        "maxlen": maxlen,
        "emb_path": "",
        "train_valid_original_file": "train_valid.csv",
        "train_valid_file": "train_valid_sequences.csv",
        "folds": folds,
        "test_original_file": "test.csv",
        "test_file": "test_sequences.csv",
        "test_window_file": "test_window_sequences.csv"
    }
    dconfig.update(other_config)
    if flag:
        dconfig["test_question_file"] = "test_question_sequences.csv"
        dconfig["test_question_window_file"] = "test_question_window_sequences.csv"

    # load old config
    if os.path.exists(configf):
        with open(configf) as fin:
            read_text = fin.read()
            if read_text.strip() == "":
                data_config = {dataset_name: dconfig}
            else:
                try:
                    data_config = json.loads(read_text)
                    if dataset_name in data_config:
                        data_config[dataset_name].update(dconfig)
                    else:
                        data_config[dataset_name] = dconfig
                except:
                    data_config = {dataset_name: dconfig}
    else:
        data_config = {dataset_name: dconfig}

    with open(configf, "w") as fout:
        data = json.dumps(data_config, ensure_ascii=False, indent=4)
        fout.write(data)

def calStatistics(df, stares, key):
    allin, allselect = 0, 0
    allqs, allcs = set(), set()
    for i, row in df.iterrows():
        rs = row["responses"].split(",")
        curlen = len(rs) - rs.count("-1")
        allin += curlen
        if "selectmasks" in row:
            ss = row["selectmasks"].split(",")
            slen = ss.count("1")
            allselect += slen
        if "concepts" in row:
            cs = row["concepts"].split(",")
            fc = list()
            for c in cs:
                cc = c.split("_")
                fc.extend(cc)
            curcs = set(fc) - {"-1"}
            allcs |= curcs
        if "questions" in row:
            qs = row["questions"].split(",")
            curqs = set(qs) - {"-1"}
            allqs |= curqs
    stares.append(",".join([str(s)
                  for s in [key, allin, df.shape[0], allselect]]))
    return allin, allselect, len(allqs), len(allcs), df.shape[0]

def get_max_concepts(df):
    max_concepts = 1
    for i, row in df.iterrows():
        cs = row["concepts"].split(",")
        num_concepts = max([len(c.split("_")) for c in cs])
        if num_concepts >= max_concepts:
            max_concepts = num_concepts
    return max_concepts

# ==============================================================================
# SECTION 3: CONCEPT SPLIT LOGIC (Main from split_datasets.py)
# ==============================================================================

def split_concept(dname, fname, dataset_name, configf, min_seq_len=3, maxlen=200, kfold=5):
    stares = []

    total_df, effective_keys = read_data(fname)
    # cal max_concepts
    if 'concepts' in effective_keys:
        max_concepts = get_max_concepts(total_df)
    else:
        max_concepts = -1

    oris, _, qs, cs, seqnum = calStatistics(total_df, stares, "original")
    print("="*20)
    print(
        f"original total interactions: {oris}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

    total_df, effective_keys = extend_multi_concepts(total_df, effective_keys)
    total_df, dkeyid2idx = id_mapping(total_df)
    dkeyid2idx["max_concepts"] = max_concepts

    extends, _, qs, cs, seqnum = calStatistics(
        total_df, stares, "extend multi")
    print("="*20)
    print(
        f"after extend multi, total interactions: {extends}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

    save_id2idx(dkeyid2idx, os.path.join(dname, "keyid2idx.json"))
    effective_keys.add("fold")
    config = []
    for key in ALL_KEYS:
        if key in effective_keys:
            config.append(key)
    # train test split & generate sequences
    train_df, test_df = train_test_split(total_df, 0.2)
    splitdf = KFold_split(train_df, kfold)
    
    splitdf[config].to_csv(os.path.join(dname, "train_valid.csv"), index=None)
    ins, ss, qs, cs, seqnum = calStatistics(
        splitdf, stares, "original train+valid")
    print(
        f"train+valid original interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    split_seqs = generate_sequences(
        splitdf, effective_keys, min_seq_len, maxlen)
    ins, ss, qs, cs, seqnum = calStatistics(
        split_seqs, stares, "train+valid sequences")
    print(
        f"train+valid sequences interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    split_seqs.to_csv(os.path.join(
        dname, "train_valid_sequences.csv"), index=None)

    # add default fold -1 to test!
    test_df["fold"] = [-1] * test_df.shape[0]
    test_df['cidxs'] = get_inter_qidx(test_df)  # add index
    test_seqs = generate_sequences(test_df, list(
        effective_keys) + ['cidxs'], min_seq_len, maxlen)
    ins, ss, qs, cs, seqnum = calStatistics(test_df, stares, "test original")
    print(
        f"original test interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    ins, ss, qs, cs, seqnum = calStatistics(
        test_seqs, stares, "test sequences")
    print(
        f"test sequences interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    print("="*20)

    test_window_seqs = generate_window_sequences(
        test_df, list(effective_keys) + ['cidxs'], maxlen)
    flag, test_question_seqs = generate_question_sequences(
        test_df, effective_keys, False, min_seq_len, maxlen)
    flag, test_question_window_seqs = generate_question_sequences(
        test_df, effective_keys, True, min_seq_len, maxlen)

    test_df = test_df[config+['cidxs']]

    test_df.to_csv(os.path.join(dname, "test.csv"), index=None)
    test_seqs.to_csv(os.path.join(dname, "test_sequences.csv"), index=None)
    test_window_seqs.to_csv(os.path.join(
        dname, "test_window_sequences.csv"), index=None)

    ins, ss, qs, cs, seqnum = calStatistics(
        test_window_seqs, stares, "test window")
    print(
        f"test window interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

    if flag:
        test_question_seqs.to_csv(os.path.join(
            dname, "test_question_sequences.csv"), index=None)
        test_question_window_seqs.to_csv(os.path.join(
            dname, "test_question_window_sequences.csv"), index=None)

        ins, ss, qs, cs, seqnum = calStatistics(
            test_question_seqs, stares, "test question")
        print(
            f"test question interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
        ins, ss, qs, cs, seqnum = calStatistics(
            test_question_window_seqs, stares, "test question window")
        print(
            f"test question window interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

    write_config(dataset_name=dataset_name, dkeyid2idx=dkeyid2idx, effective_keys=effective_keys,
                 configf=configf, dpath=dname, k=kfold, min_seq_len=min_seq_len, maxlen=maxlen, flag=flag)

    print("="*20)
    print("\n".join(stares))

# ==============================================================================
# SECTION 4: QUESTION SPLIT UTILITIES (Overrides from split_datasets_que.py)
# ==============================================================================
# Renamed with _que suffix to avoid conflicts with global functions

def generate_sequences_que(df, effective_keys, min_seq_len=3, maxlen=200, pad_val=-1):
    save_keys = list(effective_keys) + ["selectmasks"]
    dres = {"selectmasks": []}
    dropnum = 0
    for i, row in df.iterrows():
        dcur = save_dcur(row, effective_keys)

        rest, lenrs = len(dcur["responses"]), len(dcur["responses"])
        j = 0
        while lenrs >= j + maxlen:
            rest = rest - (maxlen)
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    dres[key].append(",".join(dcur[key][j: j + maxlen]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(",".join(["1"] * maxlen))

            j += maxlen
        if rest < min_seq_len:
            dropnum += rest
            continue

        pad_dim = maxlen - rest
        for key in effective_keys:
            dres.setdefault(key, [])
            if key not in ONE_KEYS:
                paded_info = np.concatenate([dcur[key][j:], np.array([pad_val] * pad_dim)])
                dres[key].append(",".join([str(k) for k in paded_info]))
            else:
                dres[key].append(dcur[key])
        dres["selectmasks"].append(",".join(["1"] * rest + [str(pad_val)] * pad_dim))

    dfinal = dict()
    for key in ALL_KEYS:
        if key in save_keys:
            dfinal[key] = dres[key]
    finaldf = pd.DataFrame(dfinal)
    print(f"dropnum: {dropnum}")
    return finaldf

def generate_window_sequences_que(df, effective_keys, maxlen=200, pad_val=-1):
    save_keys = list(effective_keys) + ["selectmasks"]
    dres = {"selectmasks": []}
    for i, row in df.iterrows():
        dcur = save_dcur(row, effective_keys)
        lenrs = len(dcur["responses"])
        if lenrs > maxlen:
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    dres[key].append(",".join(dcur[key][0: maxlen]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(",".join(["1"] * maxlen))
            for j in range(maxlen+1, lenrs+1):
                for key in effective_keys:
                    dres.setdefault(key, [])
                    if key not in ONE_KEYS:
                        dres[key].append(",".join([str(k) for k in dcur[key][j-maxlen: j]]))
                    else:
                        dres[key].append(dcur[key])
                dres["selectmasks"].append(",".join([str(pad_val)] * (maxlen - 1) + ["1"]))
        else:
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    pad_dim = maxlen - lenrs
                    paded_info = np.concatenate([dcur[key][0:], np.array([pad_val] * pad_dim)])
                    dres[key].append(",".join([str(k) for k in paded_info]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(",".join(["1"] * lenrs + [str(pad_val)] * pad_dim))

    dfinal = dict()
    for key in ALL_KEYS:
        if key in save_keys:
            dfinal[key] = dres[key]
    finaldf = pd.DataFrame(dfinal)
    return finaldf

def id_mapping_que(df):
    """
    Modified: Includes 'type' mapping for question-level processing.
    """
    id_keys = ["questions", "concepts","uid", "type"] # Added type
    dres = dict()
    dkeyid2idx = dict()
    print(f"df.columns: {df.columns}")
    for key in df.columns:
        if key not in id_keys:
            dres[key] = df[key]
    for i, row in df.iterrows():
        for key in id_keys:
            if key not in df.columns:
                continue
            dkeyid2idx.setdefault(key, dict())
            dres.setdefault(key, [])
            curids = []
            val_str = str(row[key]) if pd.notna(row[key]) else ""
            for id in val_str.split(","):
                # Handle sub-concept splits if existing (e.g. concept1_concept2)
                sub_ids = id.split('_')
                sub_curids = []
                for sub_id in sub_ids:
                    if sub_id == "": continue
                    if sub_id not in dkeyid2idx[key]:
                        dkeyid2idx[key][sub_id] = len(dkeyid2idx[key])
                    sub_curids.append(str(dkeyid2idx[key][sub_id]))
                curids.append("_".join(sub_curids))
            dres[key].append(",".join(curids))
    finaldf = pd.DataFrame(dres)
    return finaldf, dkeyid2idx


# ==============================================================================
# SECTION 5: QUESTION SPLIT LOGIC (Main from split_datasets_que.py)
# ==============================================================================

def split_question(dname, fname, dataset_name, configf, min_seq_len=3, maxlen=200, kfold=5):
    stares = []

    total_df, effective_keys = read_data(fname)
    #cal max_concepts
    if 'concepts' in effective_keys:
        max_concepts = get_max_concepts(total_df)
    else:
        max_concepts = -1

    oris, _, qs, cs, seqnum = calStatistics(total_df, stares, "original")
    print("="*20)
    print(f"original total interactions: {oris}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

     # just for id map
    total_df, dkeyid2idx = id_mapping_que(total_df)
    dkeyid2idx["max_concepts"] = max_concepts

    save_id2idx(dkeyid2idx, os.path.join(dname, "keyid2idx.json"))
    effective_keys.add("fold")

    df_save_keys = []
    for key in ALL_KEYS:
        if key in effective_keys:
            df_save_keys.append(key)

    # train test split
    train_df, test_df = train_test_split(total_df, 0.2)
    splitdf = KFold_split(train_df, kfold)
    splitdf[df_save_keys].to_csv(os.path.join(dname, "train_valid_quelevel.csv"), index=None)
    ins, ss, qs, cs, seqnum = calStatistics(splitdf, stares, "original train+valid question level")
    print(f"train+valid original interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

    # generate sequences
    split_seqs = generate_sequences_que(splitdf, effective_keys, min_seq_len, maxlen)
    ins, ss, qs, cs, seqnum = calStatistics(split_seqs, stares, "train+valid sequences question level")
    print(f"train+valid sequences interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    split_seqs.to_csv(os.path.join(dname, "train_valid_sequences_quelevel.csv"), index=None)


    # for test dataset
    # add default fold -1 to test!
    test_df["fold"] = [-1] * test_df.shape[0]
    test_seqs = generate_sequences_que(test_df, list(effective_keys), min_seq_len, maxlen)
    ins, ss, qs, cs, seqnum = calStatistics(test_df, stares, "test original question level")
    print(f"original test interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    ins, ss, qs, cs, seqnum = calStatistics(test_seqs, stares, "test sequences question level")
    print(f"test sequences interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    print("="*20)

    test_window_seqs = generate_window_sequences_que(test_df, list(effective_keys), maxlen)

    test_df = test_df[df_save_keys]
    test_df.to_csv(os.path.join(dname, "test_quelevel.csv"), index=None)
    test_seqs.to_csv(os.path.join(dname, "test_sequences_quelevel.csv"), index=None)
    test_window_seqs.to_csv(os.path.join(dname, "test_window_sequences_quelevel.csv"), index=None)

    ins, ss, qs, cs, seqnum = calStatistics(test_window_seqs, stares, "test window question level")
    print(f"test window interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

    other_config = {
        "train_valid_original_file_quelevel": "train_valid_quelevel.csv",
        "train_valid_file_quelevel": "train_valid_sequences_quelevel.csv",
        "test_file_quelevel": "test_sequences_quelevel.csv",
        "test_window_file_quelevel": "test_window_sequences_quelevel.csv",
        "test_original_file_quelevel": "test_quelevel.csv"
    }

    write_config(dataset_name=dataset_name, dkeyid2idx=dkeyid2idx, effective_keys=effective_keys,
                configf=configf, dpath = dname, k=kfold,min_seq_len = min_seq_len, maxlen=maxlen,other_config=other_config)

    print("="*20)
    print("\n".join(stares))

# ==============================================================================
# SECTION 6: RAW DATA PROCESSING & ENTRY POINT
# ==============================================================================

def process_raw_data(dataset_name, dname2paths):
    readf = dname2paths[dataset_name]
    dname = "/".join(readf.split("/")[0:-1])
    writef = os.path.join(dname, "data.txt")
    print(f"Start preprocessing data: {dataset_name}")
    
    if dataset_name == "assist2009":
        process_assist2009(readf, writef)
    elif dataset_name == "assist2017":
        process_assist2017(readf, writef)
    elif dataset_name == "aaai2023":
        dq2t = load_q2c(dname2paths["aaai2023ques"])
        process_aaai2023(readf, writef, dq2t)

    else:
        # Default fallback
        print(f"No specific processor found for {dataset_name}, ensure logic is implemented.")

    return dname, writef



def compute_assist2009_avg_rt(source_path):

    output_path = os.path.join(os.path.dirname(source_path), "questions_avgusetimes.csv")
    print(f"[Info] Calculating and discretizing average response time")
    print(f"       Source: {source_path}")

    try:
        try:
            df = pd.read_csv(source_path, usecols=['problem_id', 'ms_first_response'], encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(source_path, usecols=['problem_id', 'ms_first_response'], encoding='latin-1', low_memory=False)

        df = df.dropna(subset=['problem_id', 'ms_first_response'])
        df['ms_first_response'] = pd.to_numeric(df['ms_first_response'], errors='coerce')
        df = df.dropna(subset=['ms_first_response'])
        df = df[df['ms_first_response'] > 0]

        avg_rt_series = df.groupby('problem_id')['ms_first_response'].mean()

        problem_ids = avg_rt_series.index.astype(str).tolist()
        raw_values = avg_rt_series.values

        print("       [Process] Applying Log transformation and K-Means (k=20)...")
        log_values = np.log(raw_values)
        n_clusters = 20
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(log_values.reshape(-1, 1))

        cluster_centers = kmeans.cluster_centers_.flatten()
        sorted_indices = np.argsort(cluster_centers)
        label_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
        original_labels = kmeans.labels_
        final_labels = [label_map[l] for l in original_labels]

        qus_keys = problem_ids
        qus_vals = [str(l) for l in final_labels]

        with open(output_path, 'w', newline='', encoding="UTF8") as f:
            writer = csv.writer(f)
            writer.writerow(qus_keys)
            writer.writerow(qus_vals)

        print(f"       [Success] Saved discretized times to: {output_path}")

    except Exception as e:
        print(f"       [Error] Failed to compute average response times: {e}")
        import traceback
        traceback.print_exc()


def compute_assist2017_avg_rt(source_path):

    output_path = os.path.join(os.path.dirname(source_path), "questions_avgusetimes.csv")
    print(f"[Info] Calculating and discretizing average response time")
    print(f"       Source: {source_path}")

    try:
        try:
            df = pd.read_csv(source_path, usecols=['problemId', 'timeTaken'], encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(source_path, usecols=['problemId', 'timeTaken'], encoding='latin-1', low_memory=False)

        df = df.dropna(subset=['problemId', 'timeTaken'])
        df['timeTaken'] = pd.to_numeric(df['timeTaken'], errors='coerce')
        df = df.dropna(subset=['timeTaken'])
        df = df[df['timeTaken'] > 0]

        avg_rt_series = df.groupby('problemId')['timeTaken'].mean()

        problem_ids = avg_rt_series.index.astype(str).tolist()
        raw_values = avg_rt_series.values

        print("       [Process] Applying Log transformation and K-Means (k=20)...")
        log_values = np.log(raw_values)
        n_clusters = 20
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(log_values.reshape(-1, 1))

        cluster_centers = kmeans.cluster_centers_.flatten()
        sorted_indices = np.argsort(cluster_centers)
        label_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
        original_labels = kmeans.labels_
        final_labels = [label_map[l] for l in original_labels]

        qus_keys = problem_ids
        qus_vals = [str(l) for l in final_labels]

        with open(output_path, 'w', newline='', encoding="UTF8") as f:
            writer = csv.writer(f)
            writer.writerow(qus_keys)
            writer.writerow(qus_vals)

        print(f"       [Success] Saved discretized times to: {output_path}")

    except Exception as e:
        print(f"       [Error] Failed to compute average response times: {e}")
        import traceback
        traceback.print_exc()

def compute_aaai2023_avg_rt(source_path):

    output_path = os.path.join(os.path.dirname(source_path), "questions_avgusetimes.csv")
    print(f"[Info] Calculating and discretizing average response time")
    print(f"       Source: {source_path}")

    try:
        try:
            df = pd.read_csv(source_path, usecols=['question_id', 'response_time'], encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(source_path, usecols=['question_id', 'response_time'], encoding='latin-1', low_memory=False)

        df = df.dropna(subset=['question_id', 'response_time'])
        df['response_time'] = pd.to_numeric(df['response_time'], errors='coerce')
        df = df.dropna(subset=['response_time'])
        df = df[df['response_time'] > 0]

        avg_rt_series = df.groupby('question_id')['response_time'].mean()

        problem_ids = avg_rt_series.index.astype(str).tolist()
        raw_values = avg_rt_series.values

        print("       [Process] Applying Log transformation and K-Means (k=20)...")
        log_values = np.log(raw_values)
        n_clusters = 20
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(log_values.reshape(-1, 1))

        cluster_centers = kmeans.cluster_centers_.flatten()
        sorted_indices = np.argsort(cluster_centers)
        label_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
        original_labels = kmeans.labels_
        final_labels = [label_map[l] for l in original_labels]

        qus_keys = problem_ids
        qus_vals = [str(l) for l in final_labels]

        with open(output_path, 'w', newline='', encoding="UTF8") as f:
            writer = csv.writer(f)
            writer.writerow(qus_keys)
            writer.writerow(qus_vals)

        print(f"       [Success] Saved discretized times to: {output_path}")

    except Exception as e:
        print(f"       [Error] Failed to compute average response times: {e}")
        import traceback
        traceback.print_exc()


def process_sequences(input_file, output_file):
    print(f"正在读取文件: {input_file} ...")
    
    # 读取 CSV 文件
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        return

    # 定义包含序列数据的列名
    sequence_columns = ['questions', 'concepts', 'responses', 'timestamps', 'selectmasks', 'is_repeat']
    
    # 1. 预处理：将字符串格式的序列转换为真正的 Python 列表
    print("正在解析序列数据...")
    for col in sequence_columns:
        # 假设数据是以逗号分隔的字符串
        df[col] = df[col].apply(lambda x: str(x).split(','))

    # 2. 展开数据 (Explode)
    print("正在展开序列 (这可能需要一点时间)...")
    df_exploded = df.explode(sequence_columns)

    # 3. 类型转换
    # 展开后数据变成了字符串，需要转换为数值以便进行筛选
    df_exploded['selectmasks'] = pd.to_numeric(df_exploded['selectmasks'])
    df_exploded['is_repeat'] = pd.to_numeric(df_exploded['is_repeat'])
    # 转换 timestamp 为数值，以便后续计算 (如果包含非数值字符需要处理，这里假设是纯数字)
    df_exploded['timestamps'] = pd.to_numeric(df_exploded['timestamps'])

    # 4. 数据筛选
    print("正在筛选数据...")
    # 条件1: 删除 selectmasks 为 -1 的数据 (这是填充数据/Padding)
    # 条件2: 删除 is_repeat 为 1 的数据 (这是同一个问题的重复知识点扩展)
  #  df_filtered = df_exploded[
   #     (df_exploded['selectmasks'] != -1) & (df_exploded['is_repeat'] != 1)
   #  ].copy()
    df_filtered = df_exploded[
        (df_exploded['selectmasks'] != -1) ].copy() 


    # 5. 计算答题时间 (Response Time)
    print("正在计算答题时间...")
    # 确保数据按用户和时间顺序排列，防止不同 fold 导致的乱序
    df_filtered = df_filtered.sort_values(by=['uid', 'timestamps'])
    
    # 计算时间差：当前时间戳 - 上一次互动的时间戳
    # diff() 会计算当前行与上一行的差值，groupby 保证只在同一个用户内计算
    df_filtered['response_time'] = df_filtered.groupby('uid')['timestamps'].diff()
    
    # 第一条记录的 diff 为 NaN，填充为 0 (或者根据需求填充其他值)
    df_filtered['response_time'] = df_filtered['response_time'].fillna(0)

    # 6. 整理最终列
    final_columns = ['fold', 'uid', 'questions', 'concepts', 'responses', 'timestamps', 'response_time']
    df_final = df_filtered[final_columns]

    # 重命名列使其更具可读性
    df_final = df_final.rename(columns={
        'questions': 'question_id',
        'concepts': 'concept_id',
        'timestamps': 'timestamp'
    })

    # 7. 保存结果
    print(f"正在保存处理后的数据到: {output_file} ...")
    df_final.to_csv(output_file, index=False)
    print("处理完成！")
    print(f"原数据行数: {len(df)}")
    print(f"展开并筛选后的数据行数: {len(df_final)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset_name", type=str, default="ednet")
    parser.add_argument("-m","--min_seq_len", type=int, default=3)
    parser.add_argument("-l","--maxlen", type=int, default=200)
    parser.add_argument("-k","--kfold", type=int, default=5)
    args = parser.parse_args()

    print(args)
    dname2paths = {
        "assist2009": "../autodl-tmp/data/assist2009/skill_builder_data_corrected_collapsed.csv",
        "assist2017": "../autodl-tmp/data/assist2017/anonymized_full_release_competition_dataset.csv",
        "aaai2023_pre": "../autodl-tmp/data/aaai2023/train_valid_sequences_original.csv",
        "aaai2023": "../autodl-tmp/data/aaai2023/processed_interactions.csv",
        "aaai2023ques": "../autodl-tmp/data/aaai2023/questions.json"
    }
    configf = "../configs/data_config.json"

    if args.dataset_name == "assist2009":
        compute_assist2009_avg_rt(dname2paths["assist2009"])
    if args.dataset_name == "assist2017":
        compute_assist2017_avg_rt(dname2paths["assist2017"])
    if args.dataset_name == "aaai2023":
        process_sequences(dname2paths["aaai2023_pre"], dname2paths["aaai2023"])
        compute_aaai2023_avg_rt(dname2paths["aaai2023"])

    dname, writef = process_raw_data(args.dataset_name, dname2paths)
    print("-"*50)
    print(f"dname: {dname}, writef: {writef}")
    
    os.system("rm " + dname + "/*.pkl")

    split_concept(dname, writef, args.dataset_name, configf, args.min_seq_len,args.maxlen, args.kfold)
    print("="*100)

    split_question(dname, writef, args.dataset_name, configf, args.min_seq_len,args.maxlen, args.kfold)