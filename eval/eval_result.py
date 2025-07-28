import json
import os
import pickle
import re
import math
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, auc


RESULT_DIR = '../results/large/RQ2'

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

def get_llm_pred(file_path):
    # 加载数据
    # 新版本
    data = load_json_data(file_path)
    # 将 add_lines 转换为连续序号，并合并 add_buggy
    # 判断真实结果是否有缺陷
    real_defective = is_real_defective(data)
    llm_data = data[-1]['llm_result']
    if isinstance(llm_data, str):
        if "\"is_defect_commit\": \"YES\"" in llm_data:
            llm_data = {
                "is_defect_commit": "YES",
                "defect_lines": []
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
    model_defective = (llm_data["is_defect_commit"] == "YES")
    if "defect_lines" in llm_data:
        defect_lines = llm_data['defect_lines']
    else:
        defect_lines = []
    return real_defective, model_defective, defect_lines

def funCStatistics(file_path):
    from datetime import datetime
    data = load_json_data(file_path)
    log_data = data[-1]
    log_record = log_data['log_record']
    Step_Record = 0
    Count_Record = 0
    Fun_C_Err = 0
    Time_sp = 0
    for log in log_record:
        if log['type'] == 'Function Calling Error':
            Fun_C_Err = 1
        elif log['type'] == 'Step Record':
            message = log['message']
            Step_Record = int(message.split()[1])
        elif log['type'] == 'Count Record':
            message = log['message']
            Count_Record = int(message.split()[3])
        elif log['type'] == 'Time Record':
            message = log['message']
            start_time_str, end_time_str = message.split(" TO ")
            start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
            Time_sp = (end_time - start_time).total_seconds()
            # print(Time_sp)

    return Step_Record, Count_Record, Fun_C_Err, Time_sp

def get_add_and_del_lines(full_file):
    with open(full_file,'r',encoding='utf-8') as file:
        data = json.load(file)
    commit_info = data[3]
    # commit_info = data[1]
    content = commit_info['content']
    # 正则表达式匹配 "Changes" 后的增减数字（带符号）
    pattern = r"Changes:\s*([+-]?\d+),\s*([+-]?\d+)"
    matches = re.findall(pattern, content)

    # 初始化总数
    total_added = 0
    total_removed = 0

    # 遍历所有匹配结果并计算总数
    for added, removed in matches:
        total_added += int(added.lstrip("+-"))  # 去掉符号并累加
        total_removed += int(removed.lstrip("+-"))  # 去掉符号并累加

    # 输出结果
    # print(f"Total Added: {total_added}")
    # print(f"Total Removed: {total_removed}")

    return total_added, total_removed

def get_commit_hash(file_path):
    data = load_json_data(file_path)
    text = data[1]['content']
    # 定义匹配模式：前后文 + commit hash
    contextual_pattern = r"For the project '[^']+', its commit '([a-f0-9]{40})'"
    commit_hash = re.findall(contextual_pattern, text)[0]  # 找到当前文本中的所有符合上下文的commit hash
    return commit_hash

def convert_to_continuous_index(file_path):
    """将 add_lines 转换为从 0 开始的连续序号，并合并 add_buggy"""
    data = load_json_data(file_path)
    continuous_index_map = {}  # 记录每个 path 的 add_lines 到连续序号的映射
    continuous_add_buggy = []  # 合并后的 add_buggy 序号列表
    current_index = 0  # 当前连续序号
    paths = []
    # 遍历所有 path，构建连续序号映射
    for item in data:
        if 'path' in item:
            path = item['path']
            paths.append(path)
            add_lines = item['add_lines']
            add_buggy = item['add_buggy']
            bad = ['}', '{', ""]
            add_buggy = list(filter(lambda x: x not in bad, add_buggy))
            # 构建当前 path 的 add_lines 到连续序号的映射
            index_map = {line: idx + current_index for idx, line in enumerate(add_lines)}
            continuous_index_map[path] = index_map
            # 将 add_buggy 转换为连续序号
            continuous_add_buggy.extend([index_map[line] for line in add_buggy])
            # 更新当前连续序号
            current_index += len(add_lines)
    return continuous_index_map, continuous_add_buggy, current_index, paths

def calculate_score_for_model_result(model_result, continuous_index_map, continuous_add_buggy, total_lines):
    """对单个 path 计算分数并排序"""
    # 检查模型结果的 path 是否在连续序号映射中
    if model_result['path'] not in continuous_index_map:
        # raise ValueError(f"模型结果的 path '{model_result['path']}' 不匹配任何真实结果的 path")
        print(f"模型结果的 path '{model_result['path']}' 不匹配任何真实结果的 path")
        return []
    # 获取当前 path 的连续序号映射
    index_map = continuous_index_map[model_result['path']]

    # 剔除模型结果中不在 add_lines 中的元素
    if "suspicious" in model_result:
        if not model_result['suspicious']:
            model_result['suspicious'] = []
            valid_suspicious = []
        else:
            try:
                valid_suspicious = [index_map[line] for line in model_result['suspicious'] if line in index_map]
            except:
                print(model_result['suspicious'] )
                valid_suspicious = []
    else:
        valid_suspicious = []
    if "defective" in model_result:
        if not model_result['defective']:
            model_result['defective'] = []
            valid_defective = []
        else:
            try:
                valid_defective = [index_map[line] for line in model_result['defective'] if line in index_map]
            except:
                print(model_result['defective'])
                valid_defective = []
    else:
        valid_defective = []

    # 计算分数
    score_map = {}
    for line in valid_defective:
        score_map[line] = score_map.get(line, 0) + 4
    for line in valid_suspicious:
        # if line in valid_defective:
        #     continue
        score_map[line] = score_map.get(line, 0) + 1

    # 将所有 add_lines 的元素包含在结果中，即使分数为 0
    for line in range(total_lines):
        if line not in score_map:
            score_map[line] = 0
    # 按分数降序排序，同分时保持初始顺序
    sorted_lines = sorted(score_map.keys(), key=lambda x: (-score_map[x], x))

    # 标注真实结果中是否有相同元素
    result = []
    for line in sorted_lines:
        result.append({
            'line': line,
            'score': score_map[line],
            'is_real_buggy': line in continuous_add_buggy
        })
    return result

def merge_and_sort_results(*results):
    merged = {}  # 记录每个 line 的合并结果
    first_occurrence = {}  # 记录每个 line 第一次出现的顺序
    # 合并列表并累加 score
    for idx, result in enumerate(results):
        if not result:  # 如果列表为空，跳过
            continue
        for item in result:
            line = item['line']
            score = item['score']
            is_real_buggy = item['is_real_buggy']

            if line in merged:
                merged[line]['score'] += score
                # 如果当前 is_real_buggy 为 True，则覆盖
                if is_real_buggy:
                    merged[line]['is_real_buggy'] = True
            else:
                merged[line] = {
                    'line': line,
                    'score': score,
                    'is_real_buggy': is_real_buggy
                }
                first_occurrence[line] = idx  # 记录 line 第一次出现的列表索引
    # 将 merged 转换为列表
    merged_list = list(merged.values())
    # 按 score 从高到低排序，同分时按 line 第一次出现的顺序排序
    sorted_results = sorted(merged_list, key=lambda x: (-x['score'], first_occurrence[x['line']]))
    return sorted_results

def get_line_level(merged_result,defect_lines,continuous_index_map,continuous_add_buggy,total_lines):
    new_merged_result = merged_result
    for path_defect_lines in defect_lines:
        sorted_result = calculate_score_for_model_result(path_defect_lines, continuous_index_map,
                                                         continuous_add_buggy, total_lines)
        new_merged_result = merge_and_sort_results(merged_result, sorted_result)
    return new_merged_result

def get_recall_at_k_percent_effort(percent_effort, result_df_arg, real_buggy_commits):
    cum_LOC_k_percent = (percent_effort / 100) * result_df_arg.iloc[-1]['cum_LOC']
    buggy_line_k_percent = result_df_arg[result_df_arg['cum_LOC'] <= cum_LOC_k_percent]
    buggy_commit = buggy_line_k_percent[buggy_line_k_percent['label'] == 1]
    recall_k_percent_effort = len(buggy_commit) / float(len(real_buggy_commits))

    return recall_k_percent_effort

def eval_metrics(result_df):
    pred = result_df['defective_commit_pred']
    y_test = result_df['label']
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

def get_llm_result(result_model,tag,yuzhi=1,start=0,end=4):
    print(result_model)
    result_dir = RESULT_DIR + f'/{result_model}'
    result_files = os.listdir(result_dir)
    # 获取所有轮次的简要记录的json文件夹名字
    smalls_dir = []
    for file in result_files:
        if file.split('_')[0] == 'small':
            if end >= int(file.split('_')[1]) >= start:
                smalls_dir.append(file)
    json_files = []
    # 获取每一轮种存储结果的文件名，每一轮存储的文件名都一样的。每个文件对应一个提交
    with open('features_test.pkl','rb') as f:
        features_df = pickle.load(f)

    for small in smalls_dir:
        work_dir = result_dir + f'/{small}'
        json_files = os.listdir(work_dir)
    real_tags = []
    model_tags = []
    real_fault_count = 0
    model_fault_count = 0
    right_count = 0
    guolvhou_count = 0
    loc = []
    nf = []
    commit = []
    prob = []
    new_probs = []
    mew_probs = []
    consistencys = []
    density = []
    real_density = []

    SR = []
    CR = []
    FCE = []
    TS = []
    # 遍历所有commit
    for file in json_files:
        real = 0
        model = 0
        llm_results = []
        tSR = []
        tCR = []
        tFCE = []
        tTS = []
        # 遍历该commit的所有轮次
        for small in smalls_dir:
            r = small.split('_')[1]
            # full_file = result_dir + f"/full_{r}/{file}"
            work_dir = result_dir + f'/{small}'
            file_path = f"{work_dir}/{file}"
            print("File path", file_path)
            # 获取预测结果
            real_defective, model_defective, defect_lines = get_llm_pred(file_path)
            llm_results.append(defect_lines)
            if real_defective:
                real += 1
            if model_defective:
                if not real_defective:
                    model =model*1
                    # print("Fault predict", file_path)
                model += 1

            # Step_Record, Count_Record, Fun_C_Err, Time_sp = funCStatistics(full_file)
            tSR.append(-1)
            tCR.append(-1)
            tFCE.append(-1)
            tTS.append(-1)

        guolvhou_count += 1
        # add_lines, del_lines = get_add_and_del_lines(full_file)
        # commit_hash = get_commit_hash(full_file)
        commit_hash = file.split('.')[0]
        commit.append(commit_hash)
        record = features_df.loc[features_df['commit_hash'] == commit_hash]
        add_lines = record['la'].values[0]
        del_lines = record['ld'].values[0]
        num_file = record['nf'].values[0]
        SR.append(np.mean(tSR))
        CR.append(np.mean(tCR))
        FCE.append(np.mean(tFCE))
        TS.append(np.mean(tTS))
        add_lines = float(add_lines)
        del_lines = float(del_lines)
        num_file = float(num_file)
        loc.append(add_lines + del_lines)
        nf.append(num_file)
        prob.append(model / 5)
        density.append((model / 5) / (add_lines + del_lines))
        real_density.append((real / 5) / (add_lines + del_lines))

        if real >= 1:
            real_tags.append(1)
            real_fault_count += 1
        else:
            real_tags.append(0)
        if model >= yuzhi:
            model_tags.append(1)
            model_fault_count += 1
        else:
            model_tags.append(0)
        if real_defective == model_defective:
            right_count += 1

        merged_result=[]
        #将行转化成序号字典
        continuous_index_map, continuous_add_buggy, total_lines, paths = convert_to_continuous_index(file_path)
        #获取行级别结果
        for d_lines in llm_results:
            try:
                temp_merged_result = get_line_level(merged_result,d_lines,continuous_index_map, continuous_add_buggy, total_lines)
                merged_result = temp_merged_result
            except:
                merged_result = merged_result


        first_score = merged_result[0]['score'] if merged_result else 0
        total_score = model*5
        #一致性
        consistency = first_score/total_score if total_score>0 else 0
        consistency = consistency if consistency<1 else 1
        consistency = consistency*model/5

        # 使用一致性判断llm输出是否可信，如果一致性高，则llm输出为缺陷是更让人信服的，反之，则应该认为llm输出为无缺陷更让人信服
        # 所以，根据一致性，给llm输出结果赋予权重
        r = max(consistency,1-consistency)
        r = r**1
        Reward = (consistency-0.5*model/5)
        # if model<3:
        #     consistency = 1-consistency
        print(file)
        print(commit_hash)
        print("最高分",first_score)
        print("一致性",consistency)
        print("reward",Reward)
        print("r",r)
        print("原始概率",model/5)
        n_prob = model*(1+Reward*r)/5
        # m_prob = model * (1 + Reward ) / (model + (5 - model) * (1 -  Reward))
        m_prob = model*(1+Reward*r)/(model+(5-model)*(1-(1-r)*Reward))
        # m_prob = m_prob if m_prob <= 1 else 1
        mew_prob = m_prob
        mew_probs.append(mew_prob)
        new_prob = n_prob
        new_probs.append(new_prob)
        consistencys.append(consistency)
        print("处理后概率m", mew_prob)
        print("处理后概率n",new_prob)
        print(mew_prob-new_prob)
    m_probs = mew_probs
    n_probs = new_probs

    log_df = pd.DataFrame({
        "commit_hash": commit,
        "prob": prob,
        "m_prob":m_probs,
        "n_prob":n_probs,
        "label": real_tags,
        "loc": loc,
        "consistency":consistencys,
        "nf":nf,
        "step":SR
    })
    log_filename = f'xlsx/{result_model}/{tag}/predictions.xlsx'
    log_directory = os.path.dirname(log_filename)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    log_df.to_excel(log_filename, index=False)

    commit2file = pd.DataFrame({
        'file': json_files,
        'hash': commit
    })
    commit2file_filename = f'xlsx/{result_model}/{tag}/file2commit.xlsx'
    commit2file_directory = os.path.dirname(commit2file_filename)
    if not os.path.exists(commit2file_directory):
        os.makedirs(commit2file_directory)
    commit2file.to_excel(commit2file_filename, index=False)
    return log_df, commit2file

def convert_dtype_dataframe(df, feature_name):
    df['fix'] = df['fix'].apply(lambda x: float(bool(x)))
    df = df.astype({i: 'float32' for i in feature_name})
    return df

def load_change_metrics_df(data_dir, mode='train'):
    change_metrics = pd.read_pickle(data_dir)
    feature_name = ["ns", "nd", "nf", "entropy", "la", "ld", "lt", "fix", "ndev", "age", "nuc", "exp", "rexp", "sexp"]
    change_metrics = convert_dtype_dataframe(change_metrics, feature_name)

    return change_metrics[['commit_hash'] + feature_name]

def getWeight(lp,mp,c):
    distance = lp-mp
    base = abs(distance/2)
    bbase = 2*(1-base)
    v = 0.5
    unbe = (c - lp) / (bbase)
    # if(distance>0):
    #     unbe = (c-lp)/(bbase)
    # else:
    #     unbe = (lp-c)/(bbase)
    v = v +unbe
    v = 0 if v < 0 else v
    v = 1 if v > 1 else v
    return v

def eval(llm_model, small_model, mode, RQ=1):
    if RQ == 1:
        tag = "RQ1"
        start = 100
        end = 104
        ddir = ''
        with open('rq1_1730.txt', 'r', encoding='utf-8') as f:
            rq_list = [line.strip() for line in f.readlines()]
    elif RQ == 2:
        tag = "RQ2"
        start = 1
        end = 5
        ddir = ''
        with open('rq1_1730.txt', 'r', encoding='utf-8') as f:
            rq_list = [line.strip() for line in f.readlines()]
    elif RQ == 0:
        tag = "RQ1"
        start =100
        end =104
        ddir = ''
        with open('rq1_1730.txt', 'r', encoding='utf-8') as f:
            rq_list = [line.strip() for line in f.readlines()]
    elif RQ == 3:
        tag = "RQ3"
        start = 100
        end = 104
        ddir = 'histry/'
        with open('rq3_100.txt', 'r', encoding='utf-8') as f:
            rq_list = [line.strip() for line in f.readlines()]
    score_file = f'{ddir}{small_model}_all_fault_model_scores.xlsx'
    log_filename = f'xlsx/{llm_model}/{tag}/predictions.xlsx'
    if os.path.exists(log_filename):
        llm_df = pd.read_excel(log_filename)
    else:
        llm_df, _ = get_llm_result(llm_model,tag,1,start,end)
    # llm_df, _ = get_llm_result(llm_model, tag, 1, start, end)
    model_df = pd.read_excel(f'{ddir}{small_model}_predictions.xlsx')

    model_pred = [1 if pred else 0 for pred in model_df['pred'].tolist()]

    model_df['pred'] = model_pred

    # loc = llm_df['loc'].tolist()
    # length = len(loc)
    model_df = model_df[model_df['commit_hash'].isin(rq_list)]
    llm_df = llm_df[llm_df['commit_hash'].isin(rq_list)]

    key_column = model_df.columns[0]
    print(f"model_df['{key_column}'] data type: {model_df[key_column].apply(type).unique()}")
    print(f"llm_df['{key_column}'] data type: {llm_df[key_column].apply(type).unique()}")
    merged_df = pd.merge(model_df, llm_df, on=key_column, how='outer', suffixes=('_model', '_llm'))

    # 保存结果到新的 Excel 文件
    output_file = f'xlsx/{llm_model}/{tag}/{small_model}_merged_predictions.xlsx'
    merged_df.to_excel(output_file, index=False)
    # if RQ == 0:
    #     merged_df=[merged_df['nf']<10]
    # merged_df = merged_df[2*merged_df['nf'] < merged_df['step']]
    df = merged_df
    loc = df['loc'].tolist()
    commit_hash = df['commit_hash'].tolist()
    prob_model = df['prob_model'].tolist()
    prob_llm = df['prob_llm'].tolist()
    consistency = df['consistency']
    n_prob = df['n_prob']
    m_prob = df['m_prob']
    label = df['label_model'].tolist()

    pred_model = df['pred'].tolist()

    prob = [a + b for a, b in zip(prob_llm, prob_model)]

    min_val = min(prob)
    max_val = max(prob)
    prob_normalized = [(x - min_val) / (max_val - min_val) for x in prob]
    result_prob = []
    if mode == 'vote':
        # 投票法
        pred = []
        for pm,pl in zip(pred_model, prob_llm):
            if int(pm)==1 or int(pl)==1:
                pred.append(1)
            else:
                pred.append(0)
        result_prob = prob_normalized
    elif mode == 'os':
        pred = [int(pr) for pr in pred_model]
        result_prob = prob_model
    elif mode == 'ol':
        pred = []
        for pl in m_prob:
            if pl > 0.6:
                pred.append(1)
            else:
                pred.append(0)
        result_prob = m_prob
    elif mode =='np':
        pred = []
        for n_p,pm,c in zip(n_prob, prob_model, consistency):
            v = getWeight(n_p,pm, c)
            if (v * n_p + (1 - v) * pm) >= 0.5:
                pred.append(1)
            else:
                pred.append(0)
            result_prob.append((v * n_p + (1 - v) * pm))
    elif mode =='mp':
        pred = []
        for m_p,pm, c in zip(m_prob, prob_model,consistency):
            v = getWeight(m_p,pm,c)
            # if (m_p>=0.5 and pm<0.5) or (m_p<0.5 and pm >= 0.5):
            #     v = getWeight(m_p,pm,c)
            # else:
            #     v = 0.5
            if (v*m_p+(1-v)*pm)>=0.5:
                pred.append(1)
            else:
                pred.append(0)
            result_prob.append((v*m_p+(1-v)*pm))
    else:
        # 概率法
        pred = []
        for pr in prob_normalized:
            if pr > 0.55:
                pred.append(1)
            else:
                pred.append(0)
        result_prob = prob_normalized

    result_df = pd.DataFrame({
        'defective_commit_pred': pred,
        'label': label,
        'LOC': loc,
        'defective_commit_prob': result_prob
    })

    f1, auc, recall_20_percent_effort, effort_at_20_percent_LOC_recall, p_opt = eval_metrics(result_df)
    data = {
        'commit_hash': commit_hash,
        'prob': result_prob,
        'pred': pred,
        'label': label
    }
    ddf = pd.DataFrame(data)


    record_df = pd.DataFrame({
        'commit_hash':commit_hash,
        'prob_model':prob_model,
        'prob_llm_ori':prob_llm,
        'prob_llm':m_prob,
        'consistency':consistency,
        'prob': result_prob,
        'pred': pred,
        'label': label
    })
    finial_file = f'xlsx/{llm_model}/{tag}/{small_model}_{mode}_predictions.xlsx'
    record_df.to_excel(finial_file,index=False)

    prec, rec, f11, _ = precision_recall_fscore_support(label, pred, average='binary')

    tn, fp, fn, tp = confusion_matrix(label, pred, labels=[0, 1]).ravel()


    rec = tp / (tp + fn)

    FAR = fp / (fp + tn)  # false alarm rate

    AUC = roc_auc_score(label, result_prob)

    fault_df = ddf[ddf['pred'] == 1]
    fault_df = fault_df[fault_df['label'] == 1]
    fault_hash = fault_df['commit_hash'].tolist()
    score_df = pd.read_excel(score_file)
    score_df = score_df[score_df['commit_hash'].isin(fault_hash)]

    IFAs = score_df['IFA'].tolist()
    top_10s = score_df['TOP10'].tolist()
    top_5s = score_df['TOP5'].tolist()
    top_20_percent_LOC_recalls = score_df['T20R'].tolist()
    effort_at_20_percent_LOC_recalls = score_df['E20R'].tolist()

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
    print("+" * 200)
    print("IFA", np.mean(IFAs))
    print("TOP10", np.mean(top_10s))
    print("TOP5", np.mean(top_5s))
    print("T20", np.mean(top_20_percent_LOC_recalls))
    print("E20", np.mean(effort_at_20_percent_LOC_recalls))

eval('gpt-4o-mini','jitfine','mp',2)