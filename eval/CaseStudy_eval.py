import math
import os
import re
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, auc
import numpy as np
import pandas as pd
import json

RESULT_DIR = '../results/large/RQ3/Ano'

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

def convert_to_continuous_index(data):
    """将 add_lines 转换为从 0 开始的连续序号，并合并 add_buggy"""
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
        valid_suspicious = [index_map[line] for line in model_result['suspicious'] if line in index_map]
    else:
        valid_suspicious = []
    if "defective" in model_result:
        if not model_result['defective']:
            model_result['defective'] = []
        valid_defective = [index_map[line] for line in model_result['defective'] if line in index_map]
    else:
        valid_defective = []

    # 计算分数
    score_map = {}
    for line in valid_defective:
        score_map[line] = score_map.get(line, 0) + 2
    for line in valid_suspicious:
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


def main(file_path):
    # 加载数据
    # 新版本
    data = load_json_data(file_path)
    # 将 add_lines 转换为连续序号，并合并 add_buggy
    continuous_index_map, continuous_add_buggy, total_lines, paths = convert_to_continuous_index(data)
    # 判断真实结果是否有缺陷
    real_defective = is_real_defective(data)
    # 判断模型预测结果是否有缺陷
    # 填充字符类型结果
    llm_data = data[-1]['llm_result']
    if isinstance(llm_data,str):
        if "\"defective\": [\n" in llm_data or "\"suspicious\": [\n" in llm_data:
            tianchong = "####"
        else:
            tianchong = None
        defect_lines = []
        for path in paths:
            defect_lines.append({
                "path": path,
                "suspicious": [tianchong],
                "defective": [],
                "explanation":""
            })
        llm_data = {
            "is_defect_commit": "YES",
            "defect_lines": defect_lines
        }
    if not llm_data :
        defect_lines = []
        for path in paths:
            defect_lines.append({
                "path": path,
                "suspicious": [],
                "defective": []
            })
        llm_data = {
            "is_defect_commit": "NO",
            "defect_lines": defect_lines
        }
    model_defective = (llm_data["is_defect_commit"] == "YES")
    merged_result = []
    # 提取所有 llm_result
    llm_results = llm_data['defect_lines']

    # 对每个 llm_result 计算分数并排序
    for idx, llm_result in enumerate(llm_results):
        if isinstance(llm_result, dict):
            try:
                sorted_result = calculate_score_for_model_result(llm_result, continuous_index_map,
                                                                     continuous_add_buggy, total_lines)
                merged_result = merge_and_sort_results(merged_result,sorted_result)
            except ValueError as e:
                print(f"错误1: {e}")
        elif isinstance(llm_result, list):
            for model_result in llm_result:
                try:
                    sorted_result = calculate_score_for_model_result(model_result, continuous_index_map,
                                                                         continuous_add_buggy,total_lines)
                    merged_result = merge_and_sort_results(merged_result, sorted_result)
                except ValueError as e:
                    print(f"错误2: {e}")
    return merged_result, real_defective, model_defective, len(continuous_add_buggy), len(paths), continuous_index_map


def divide_num(index,buggyline_counts):
    total = 0
    for i in range(len(buggyline_counts)):
        if i+1 <= index:
            total += (i+1)*buggyline_counts[i]
        else:
            total += (index)*buggyline_counts[i]
    return total

def split_coordinates(data):
    x_coords, y_coords = zip(*data)
    return list(x_coords), list(y_coords)

def get_add_and_del_lines(full_file):
    with open(full_file,'r',encoding='utf-8') as file:
        data = json.load(file)
    # commit_info = data[3]
    commit_info = data[0]
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

def get_recall_at_k_percent_effort(percent_effort, result_df_arg, real_buggy_commits):
    cum_LOC_k_percent = (percent_effort / 100) * result_df_arg.iloc[-1]['cum_LOC']
    buggy_line_k_percent = result_df_arg[result_df_arg['cum_LOC'] <= cum_LOC_k_percent]
    buggy_commit = buggy_line_k_percent[buggy_line_k_percent['label'] == 1]
    recall_k_percent_effort = len(buggy_commit) / float(len(real_buggy_commits))
    return recall_k_percent_effort

def map_result_to_code(merged_result, continuous_index_map):
    result_with_code = []
    for item in merged_result:
        line_index = item['line']
        for path, index_map in continuous_index_map.items():
            if line_index in index_map.values():
                # 找到对应的代码行
                code_line = list(index_map.keys())[list(index_map.values()).index(line_index)]
                result_with_code.append({
                    'seq':item['line'],
                    'path': path,
                    'code_line': code_line,
                    'score': item['score'],
                    'is_real_buggy': item['is_real_buggy']
                })
                # break
    return result_with_code

def write_map(result_with_code, excel_file_path):
    import pandas as pd
    import os
    # 检查文件是否存在
    if os.path.exists(excel_file_path):
        # 如果文件存在，读取现有数据
        existing_df = pd.read_excel(excel_file_path)
        # 将新数据转换为 DataFrame
        new_df = pd.DataFrame(result_with_code,columns=['seq', 'path', 'code_line', 'score', 'is_real_buggy'])
        # 追加新数据到现有数据
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # 如果文件不存在，直接创建新的 DataFrame
        combined_df = pd.DataFrame(result_with_code,columns=['seq', 'path', 'code_line', 'score', 'is_real_buggy'])
    # 将数据写入 Excel 文件
    combined_df.to_excel(excel_file_path, index=False)
    print(f"数据已成功写入 {excel_file_path}")

def file2hash(filename):
    # 读取 Excel 文件
    # file_path = 'commits_files.xlsx'  # 替换为你的 Excel 文件路径
    # df = pd.read_excel(file_path)
    # # 提取 file 和 hash 列的数据
    # files = df['file'].tolist()  # 将 'file' 列转换为列表
    # hashs = df['hash'].tolist()  # 将 'hash' 列转换为列表
    # json_name = filename.split('.')[0] + '.json'
    # index = files.index(json_name)
    # hash = hashs[index]
    hash = "1234567890"
    return hash

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
            print(Time_sp)

    return Step_Record, Count_Record, Fun_C_Err, Time_sp

def eval(result_model, yuzhi=1,hangjiezhi = 15,wenjianjiezhi = 4,start = 0, end =4):
    result_dir = RESULT_DIR +f'/{result_model}'
    result_files = os.listdir(result_dir)
    smalls_dir = []
    for file in result_files:
        if file.split('_')[0] == 'small':
            if end>=int(file.split('_')[1])>=start:
                smalls_dir.append(file)

    json_files = []
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
    prob = []
    density = []
    real_density = []
    top_5s = []
    top_10s = []
    IFAs = []
    top_20_percent_LOC_recalls = []
    effort_at_20_percent_LOC_recalls = []
    hashs = []
    all_hashs = []

    SR = []
    CR = []
    FCE = []
    TS = []

    cons = []

    for file in json_files:
        merged_result = []
        real = 0
        model = 0
        continue_flag = False
        for small in smalls_dir:
            r = small.split('_')[1]
            full_file = result_dir + f"/full_{r}/{file}"
            work_dir = result_dir + f'/{small}'
            file_path = f"{work_dir}/{file}"
            print("File path",file_path)
            try:
                sorted_result, real_defective, model_defective, buggy_len, path_len, index_map = main(file_path)
            except:
                continue_flag = True
                continue
            if real_defective:
                real += 1
            if model_defective:
                if not real_defective:
                    print("Fault predict",file_path)
                model += 1
            merged_result = merge_and_sort_results(merged_result,sorted_result)

            Step_Record, Count_Record, Fun_C_Err, Time_sp = funCStatistics(full_file)
            SR.append(Step_Record)
            CR.append(Count_Record)
            FCE.append(Fun_C_Err)
            TS.append(Time_sp)

        if continue_flag:
            continue
        if len(merged_result) > hangjiezhi:
            continue
        if path_len > wenjianjiezhi:
            continue
        guolvhou_count += 1

        add_lines, del_lines = get_add_and_del_lines(full_file)
        loc.append(add_lines + del_lines)
        prob.append(model/5)
        density.append((model/5)/(add_lines+del_lines))
        real_density.append((real/5)/(add_lines+del_lines))

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
        # 此处算line级别
        line_seq = [item['line'] for item in merged_result]
        line_score = [item['score'] for item in merged_result]
        line_label = [1 if item['is_real_buggy'] else 0 for item in merged_result]


        if real >= 1 and model >=yuzhi:
            # 提取 line_seq、line_score 和 line_label
            if model > 0:
                if line_score:
                    h_score = line_score[0]
                else:
                    h_score = 0
                all_score = 3 * model
                con = h_score / all_score
                cons.append(con)

            # 打印结果
            line_df = pd.DataFrame({
                'label': line_label,
                'scr': line_score
            })
            line_df['row'] = np.arange(1, len(line_df) + 1)

            real_buggy_lines = line_df[line_df['label'] == 1]

            top_10_acc = 0
            top_5_acc = 0
            if len(real_buggy_lines) < 1:
                IFA = len(line_df)
                top_20_percent_LOC_recall = 0
                effort_at_20_percent_LOC_recall = math.ceil(0.2 * len(line_df))

            else:
                IFA = line_df[line_df['label'] == 1].iloc[0]['row'] - 1
                label_list = list(line_df['label'])

                all_rows = len(label_list)

                # find top-10 accuracy
                if all_rows < 10:
                    top_10_acc = np.sum(label_list[:all_rows]) / len(label_list[:all_rows])
                else:
                    top_10_acc = np.sum(label_list[:10]) / len(label_list[:10])

                # find top-5 accuracy
                if all_rows < 5:
                    top_5_acc = np.sum(label_list[:all_rows]) / len(label_list[:all_rows])
                else:
                    top_5_acc = np.sum(label_list[:5]) / len(label_list[:5])

                # find recall
                LOC_20_percent = line_df.head(int(0.2 * len(line_df)))
                buggy_line_num = LOC_20_percent[LOC_20_percent['label'] == 1]
                top_20_percent_LOC_recall = float(len(buggy_line_num)) / float(len(real_buggy_lines))

                # find effort @20% LOC recall

                buggy_20_percent = real_buggy_lines.head(math.ceil(0.2 * len(real_buggy_lines)))
                buggy_20_percent_row_num = buggy_20_percent.iloc[-1]['row']
                effort_at_20_percent_LOC_recall = int(buggy_20_percent_row_num) / float(len(line_df))

            if top_5_acc == 0:
                print("top_5_0",file, f"有 {path_len} 个文件: {buggy_len}/{len(line_df)}")
            if top_10_acc == 0:
                print("top_10_0",file, f"有 {path_len} 个文件: {buggy_len}/{len(line_df)}")

            IFAs.append(IFA)
            top_10s.append(top_10_acc)
            top_5s.append(top_5_acc)
            top_20_percent_LOC_recalls.append(top_20_percent_LOC_recall)
            effort_at_20_percent_LOC_recalls.append(effort_at_20_percent_LOC_recall)
            hash = file2hash(file)
            hashs.append(hash)


        commit_hash = file2hash(file)
        all_hashs.append(commit_hash)




    result_df = pd.DataFrame({
        "LOC":loc,
        "defect_density":density,
        "actual_defect_density":real_density,
        "label":real_tags,
        "defective_commit_prob": prob
    })

    print(len(all_hashs))
    print(len(prob))
    print(len(real_tags))
    # relog_df = pd.DataFrame({
    #     'commit_hash': all_hashs,
    #     'prob': prob,
    #     'pred': prob,
    #     'label': real_tags
    # })
    # relog_path = f'g35t_predictions.xlsx'
    # relog_df.to_excel(relog_path, index=False)
    # print(f"数据已成功写入 {relog_path}")



    result_df = result_df.sort_values(by='defect_density', ascending=False)
    actual_result_df = result_df.sort_values(by='actual_defect_density', ascending=False)
    actual_worst_result_df = result_df.sort_values(by='actual_defect_density', ascending=True)

    result_df['cum_LOC'] = result_df['LOC'].cumsum()
    actual_result_df['cum_LOC'] = actual_result_df['LOC'].cumsum()
    actual_worst_result_df['cum_LOC'] = actual_worst_result_df['LOC'].cumsum()

    real_buggy_commits = result_df[result_df['label'] == 1]

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

    print("+" * 200)

    prec, rec, f1, _ = precision_recall_fscore_support(real_tags, model_tags, average='binary')
    tn, fp, fn, tp = confusion_matrix(real_tags, model_tags, labels=[0, 1]).ravel()

    FAR = fp / (fp + tn)  # false alarm rate

    AUC = roc_auc_score(real_tags, prob)


    print("调用链长度（平均）",np.mean(SR))
    print("FC失败结束（平均）", np.sum(FCE))
    print("FC失败次数（平均）", np.mean(CR))
    print("时长（平均）", np.mean(TS))
    print("+" * 200)
    print("tn负类预测为负类",tn)
    print("fp负类预测为正类",fp)
    print("fn正类预测为负类",fn)
    print("tp正类预测为正类",tp)
    print("prec",prec)
    print("rec",rec)
    print("f1",f1)
    print("FAR",FAR)
    print("AUC",AUC)
    print("R@20E",recall_20_percent_effort)
    print("E@20R",effort_at_20_percent_LOC_recall)
    print("POpt",p_opt)
    print("+" * 200)
    print("IFA",np.mean(IFAs))
    print("TOP10",np.mean(top_10s))
    print("TOP5",np.mean(top_5s))
    print("T20",np.mean(top_20_percent_LOC_recalls))
    print("E20",np.mean(effort_at_20_percent_LOC_recalls))
    print("Con",np.mean(cons))

def has_non_empty_array(text):
    # 方法1：直接遍历字符串（100%可靠）
    for i in range(len(text)):
        if text[i] == '[':
            # 检查 [ 后面是否有非空白、非 ] 的字符
            j = i + 1
            while j < len(text) and text[j].isspace():
                j += 1
            if j < len(text) and text[j] != ']':
                return True
    return False

def new_main(file_path):
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
    elif llm_data["defect_lines"]:
        json_str = json.dumps(llm_data["defect_lines"], indent=4, ensure_ascii=False)
        model_defective = has_non_empty_array(json_str)
        # model_defective = False
        # if isinstance(llm_data["defect_lines"],list):
        #     for path_t in llm_data["defect_lines"]:
        #         if path_t['suspicious'] or path_t['defective']:
        #             model_defective = True
        #             break
        # else:
        #     # 将 JSON 转为带缩进格式的字符串
        #     json_str = json.dumps(llm_data["defect_lines"], indent=4, ensure_ascii=False)
        #
        #     print("=== 输出 JSON 字符串 ===")
        #     print(json_str)
        #     print("=======================")
        #
        #     model_defective = has_non_empty_array(json_str)
        #     print(model_defective)

    else:
        model_defective = False

    # model_defective = ((llm_data["is_defect_commit"] == "YES") or llm_data["defect_lines"])

    return real_defective, model_defective

def new_eval(result_model, yuzhi=1,hangjiezhi = 15,wenjianjiezhi = 4,start = 0, end =4):
    result_dir = RESULT_DIR +f'/{result_model}'
    result_files = os.listdir(result_dir)
    smalls_dir = []
    for file in result_files:
        if file.split('_')[0] == 'small':
            if end>=int(file.split('_')[1])>=start:
                smalls_dir.append(file)

    json_files = []
    for small in smalls_dir:
        work_dir = result_dir + f'/{small}'
        json_files = os.listdir(work_dir)
    real_tags = []
    model_tags = []
    real_fault_count = 0
    model_fault_count = 0
    right_count = 0
    guolvhou_count = 0
    prob = []

    SR = []
    CR = []
    FCE = []
    TS = []


    for file in json_files:
        real = 0
        model = 0
        for small in smalls_dir:
            r = small.split('_')[1]
            full_file = result_dir + f"/full_{r}/{file}"
            work_dir = result_dir + f'/{small}'
            file_path = f"{work_dir}/{file}"
            # print("File path",file_path)
            real_defective, model_defective = new_main(file_path)
            if real_defective:
                real += 1
            if model_defective:
                # if not real_defective:
                    # print("Fault predict",file_path)
                model += 1

            Step_Record, Count_Record, Fun_C_Err, Time_sp = funCStatistics(full_file)
            SR.append(Step_Record)
            CR.append(Count_Record)
            FCE.append(Fun_C_Err)
            TS.append(Time_sp)
        guolvhou_count += 1

        prob.append(model/5)

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
    print(len(prob))
    print(len(real_tags))

    prec, rec, f1, _ = precision_recall_fscore_support(real_tags, model_tags, average='binary')
    tn, fp, fn, tp = confusion_matrix(real_tags, model_tags, labels=[0, 1]).ravel()

    FAR = fp / (fp + tn)  # false alarm rate

    print("调用链长度（平均）", np.mean(SR))
    print("FC失败结束（平均）", np.sum(FCE))
    print("FC失败次数（平均）", np.mean(CR))
    print("时长（平均）", np.mean(TS))
    print("+" * 200)
    print("tn负类预测为负类", tn)
    print("fp负类预测为正类", fp)
    print("fn正类预测为负类", fn)
    print("tp正类预测为正类", tp)
    print("prec", prec)
    print("rec", rec)
    print("f1", f1)
    print("FAR", FAR)

def getNofunCommitHash(full_path):
    with open(full_path,'r',encoding='utf-8') as f:
        data = json.load(f)
    content = data[0]['content']
    contextual_pattern = r"Commit Hash: ([a-f0-9]{40})"
    commit_hash = re.findall(contextual_pattern, content)[0]
    return commit_hash

def eval_no_function(result_model,yuzhi, start,end):
    result_dir = RESULT_DIR + f'/{result_model}'
    result_files = os.listdir(result_dir)
    smalls_dir = []
    for file in result_files:
        if file.split('_')[0] == 'small':
            if end >= int(file.split('_')[1]) >= start:
                smalls_dir.append(file)

    json_files = []
    for small in smalls_dir:
        work_dir = result_dir + f'/{small}'
        json_files = os.listdir(work_dir)

    with open('rq3_100.txt', 'r', encoding='utf-8') as f:
        rq_list = [line.strip() for line in f.readlines()]

    tn = 0
    fp = 0
    fn = 0
    tp = 0

    for file in json_files:
        real = 0
        model = 0
        c_flag = False
        for small in smalls_dir:
            r = small.split('_')[1]
            full_file = result_dir + f"/full_{r}/{file}"
            work_dir = result_dir + f'/{small}'
            file_path = f"{work_dir}/{file}"
            # print("File path", file_path)
            commit_hash = getNofunCommitHash(full_file)
            if commit_hash not in rq_list:
                c_flag = True
                continue
            try:
                real_defective, model_defective = new_main(file_path)
            except:
                real_defective = 0
                model_defective = 0
            if real_defective:
                real += 1
            if model_defective:
                # if not real_defective:
                    # print("Fault predict", file_path)
                model += 1
        if c_flag:
            continue
        if model>=yuzhi:
            if real>0:
                tp += 1
            else:
                fp += 1
        else:
            if real>0:
                fn += 1
            else:
                tn += 1
    print("tn负类预测为负类", tn)
    print("fp负类预测为正类", fp)
    print("fn正类预测为负类", fn)
    print("tp正类预测为正类", tp)

# eval('gpt-3.5-turbo',3,2000000,10000000,1,5)
# eval('gpt-4o-mini',3,200000,1000000,0,4)
# eval_no_function('gemini-2.0-flash',3,1,5)
new_eval('gpt-3.5-turbo',3,200000,10000000,1,5)



# # 测试用例
# case1 = """=== 输出 JSON 字符串 ===
# {
#     "path": "parquet-tools/src/main/java/org/apache/parquet/tools/Main.java",
#     "suspicious": [],
#     "defective": [],
#     "explanation": "No confirmed defective or suspicious code lines were identified in the analyzed changes."
# }
# ======================="""
#
# case2 = """=== 输出 JSON 字符串 ===
# {
#     "path": "parquet-pig/src/main/java/parquet/pig/ParquetLoader.java",
#     "suspicious": [],
#     "defective": [
#         "    setInput(storage.getStatistics(inputFile, getUDFContext().getUDFProperties()));",
#         "    PARQUET_INPUT_STORAGE_FACTORY.getStorage(path, configuration)"
#     ]
# }
# ======================="""
#
# case3 = """=== 输出 JSON 字符串 ===
# {
#     "path": "some/file.java",
#     "suspicious": ["some suspicious line"],
#     "defective": []
# }
# ======================="""
#
# print(has_non_empty_array(case1))  # False（defective 和 suspicious 都为空）
# print(has_non_empty_array(case2))  # True（defective 非空）
# print(has_non_empty_array(case3))  # True（suspicious 非空）
