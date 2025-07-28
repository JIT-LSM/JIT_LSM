import math
import os
import pickle
import re

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, auc
import numpy as np
import json

RESULT_DIR = '../results/large/RQ3/OneLLM'

def load_json_data(file_path):
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 替换所有 "File path" 为 "path"
    updated_content = content.replace("File path", "path")
    data = json.loads(updated_content)
    return data


def is_real_defective(data):
    """判断真实结果是否有缺陷（所有 add_buggy 是否都为空）"""
    for item in data:
        if 'add_buggy' in item and item['add_buggy']:
            return True
    return False

def get_recall_at_k_percent_effort(percent_effort, result_df_arg, real_buggy_commits):
    cum_LOC_k_percent = (percent_effort / 100) * result_df_arg.iloc[-1]['cum_LOC']
    buggy_line_k_percent = result_df_arg[result_df_arg['cum_LOC'] <= cum_LOC_k_percent]
    buggy_commit = buggy_line_k_percent[buggy_line_k_percent['label'] == 1]
    recall_k_percent_effort = len(buggy_commit) / float(len(real_buggy_commits))

    return recall_k_percent_effort

def eval_metrics(result_df):
    pred = result_df['defective_commit_pred']
    y_test = result_df['label']
    print(y_test)
    print(pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average='binary')  # at threshold = 0.5
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    #     rec = tp/(tp+fn)

    FAR = fp / (fp + tn)  # false alarm rate
    dist_heaven = math.sqrt((pow(1 - rec, 2) + pow(0 - FAR, 2)) / 2.0)  # distance to heaven

    AUC = roc_auc_score(y_test, result_df['defective_commit_prob'])

    result_df['defect_density'] = result_df['defective_commit_prob'] / result_df['LOC']  # predicted defect density
    result_df['actual_defect_density'] = result_df['label'] / result_df['LOC']  # defect density

    result_df = result_df.sort_values(by='defect_density', ascending=False)
    actual_result_df = result_df.sort_values(by='actual_defect_density', ascending=False)
    actual_worst_result_df = result_df.sort_values(by='actual_defect_density', ascending=True)

    result_df['cum_LOC'] = result_df['LOC'].cumsum()
    actual_result_df['cum_LOC'] = actual_result_df['LOC'].cumsum()
    actual_worst_result_df['cum_LOC'] = actual_worst_result_df['LOC'].cumsum()

    real_buggy_commits = result_df[result_df['label'] == 1]

    label_list = list(result_df['label'])

    all_rows = len(label_list)

    # find Recall@20%Effort
    cum_LOC_20_percent = 0.2 * result_df.iloc[-1]['cum_LOC']
    buggy_line_20_percent = result_df[result_df['cum_LOC'] <= cum_LOC_20_percent]
    buggy_commit = buggy_line_20_percent[buggy_line_20_percent['label'] == 1]
    recall_20_percent_effort = len(buggy_commit) / float(len(real_buggy_commits))

    # find Effort@20%Recall
    buggy_20_percent = real_buggy_commits.head(math.ceil(0.2 * len(real_buggy_commits)))
    buggy_20_percent_LOC = buggy_20_percent.iloc[-1]['cum_LOC']
    effort_at_20_percent_LOC_recall = int(buggy_20_percent_LOC) / float(result_df.iloc[-1]['cum_LOC'])

    # find P_opt
    percent_effort_list = []
    predicted_recall_at_percent_effort_list = []
    actual_recall_at_percent_effort_list = []
    actual_worst_recall_at_percent_effort_list = []

    for percent_effort in np.arange(10, 101, 10):
        predicted_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, result_df,
                                                                           real_buggy_commits)
        actual_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, actual_result_df,
                                                                        real_buggy_commits)
        actual_worst_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, actual_worst_result_df,
                                                                              real_buggy_commits)

        percent_effort_list.append(percent_effort / 100)

        predicted_recall_at_percent_effort_list.append(predicted_recall_k_percent_effort)
        actual_recall_at_percent_effort_list.append(actual_recall_k_percent_effort)
        actual_worst_recall_at_percent_effort_list.append(actual_worst_recall_k_percent_effort)

    p_opt = 1 - ((auc(percent_effort_list, actual_recall_at_percent_effort_list) -
                  auc(percent_effort_list, predicted_recall_at_percent_effort_list)) /
                 (auc(percent_effort_list, actual_recall_at_percent_effort_list) -
                  auc(percent_effort_list, actual_worst_recall_at_percent_effort_list)))

    return f1, AUC, recall_20_percent_effort, effort_at_20_percent_LOC_recall, p_opt


def check_small_file(file_path):
    data = load_json_data(file_path)
    real_defective = is_real_defective(data)
    llm_data = data[-1]['llm_result']
    if isinstance(llm_data, str):
        if "\"is_defect_commit\": \"YES\"" in llm_data:
            llm_data = {
                "is_defect_commit":"YES",
                "defect_lines": [{
                    "path": "1",
                    "suspicious": ['1'],
                    "defective": ['1']
                }]
            }
        else:
            llm_data = {
                "is_defect_commit": "NO",
                "defect_lines": []
            }
    if not llm_data:
        llm_data = {
            "is_defect_commit": "NO",
            "defect_lines": []
        }
    if (llm_data["is_defect_commit"] == "YES"):
        model_defective = True
    else:
        model_defective = False
    return real_defective, model_defective

def eval_ablation(result_model, yuzhi=1,start = 0, end =4):
    result_dir = RESULT_DIR +f'/{result_model}'
    result_files = os.listdir(result_dir)
    smalls_dir = []
    #记录文件目录数
    for file in result_files:
        if file.split('_')[0] == 'small':
            if end>=int(file.split('_')[1])>=start:
                smalls_dir.append(file)
    json_files = []
    #记录文件数
    for small in smalls_dir:
        work_dir = result_dir + f'/{small}'
        json_files = os.listdir(work_dir)

    #获取特征信息
    with open('features_test.pkl','rb') as f:
        features_df = pickle.load(f)

    real_tags = []
    model_tags = []
    real_fault_count = 0
    model_fault_count = 0
    right_count = 0
    guolvhou_count = 0
    prob = []
    loc = []

    #遍历文件
    for file in json_files:
        real = 0
        model = 0
        for small in smalls_dir:
            r = small.split('_')[1]
            # full_file = result_dir + f"/full_{r}/{file}"
            work_dir = result_dir + f'/{small}'
            file_path = f"{work_dir}/{file}"
            real_defective, model_defective = check_small_file(file_path)
            if real_defective:
                real += 1
            if model_defective:
                model += 1
        guolvhou_count += 1

        prob.append(model/5)
        commit_hash = file.split('.')[0]

        if real >= 1:
            real_tags.append(1)
            real_fault_count += 1
        else:
            real_tags.append(0)
        if model >=yuzhi:
            model_tags.append(1)
            model_fault_count += 1
        else:
            model_tags.append(0)
        if real_defective == model_defective:
            right_count+=1

        record = features_df.loc[features_df['commit_hash'] == commit_hash]
        add_lines = record['la'].values[0]
        del_lines = record['ld'].values[0]
        add_lines = float(add_lines)
        del_lines = float(del_lines)
        loc.append(add_lines + del_lines)



    print(len(prob))
    print(len(real_tags))

    pred = model_tags
    label = real_tags
    result_prob = prob


    result_df = pd.DataFrame({
        'defective_commit_pred': pred,
        'label': label,
        'LOC': loc,
        'defective_commit_prob': result_prob
    })

    f1, auc, recall_20_percent_effort, effort_at_20_percent_LOC_recall, p_opt = eval_metrics(result_df)

    prec, rec, f1, _ = precision_recall_fscore_support(real_tags, model_tags, average='binary')
    tn, fp, fn, tp = confusion_matrix(real_tags, model_tags, labels=[0, 1]).ravel()

    FAR = fp / (fp + tn)  # false alarm rate

    AUC = roc_auc_score(label, result_prob)

    print("+" * 200)
    print("tn负类预测为负类", tn)
    print("fp负类预测为正类", fp)
    print("fn正类预测为负类", fn)
    print("tp正类预测为正类", tp)
    print("prec", prec)
    print("rec", rec)
    print("f1", f1)
    print("FAR", FAR)
    print("AUC", AUC)
    print("R@20E", recall_20_percent_effort)
    print("E@20R", effort_at_20_percent_LOC_recall)
    print("POpt", p_opt)

def getNofunCommitHash(full_path):
    with open(full_path,'r',encoding='utf-8') as f:
        data = json.load(f)
    content = data[0]['content']
    contextual_pattern = r"Commit Hash: ([a-f0-9]{40})"
    commit_hash = re.findall(contextual_pattern, content)[0]
    return commit_hash


eval_ablation('gpt-3.5-turbo',3,1,5)
