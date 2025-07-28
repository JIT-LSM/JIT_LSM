import json
import os
import re
from pathlib import Path


def load_json_data(file_path):
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 替换所有 "File path" 为 "path"
    updated_content = content.replace("File path", "path")
    data = json.loads(updated_content)
    return data

def new_main(file_path):
    data = load_json_data(file_path)
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
    json_str = json.dumps(llm_data["defect_lines"], indent=4, ensure_ascii=False)
    return json_str

def count_suspicious(text):
    # 匹配完全的："suspicious": []
    pattern1 = r'"suspicious":\s*\[\s*\]'
    count1 = len(re.findall(pattern1, text))

    # 匹配单独的："suspicious"
    pattern2 = r'"suspicious": \['
    count2 = len(re.findall(pattern2, text))
    return count1,count2

def count_defective(text):
    # 匹配完全的："suspicious": []
    pattern1 = r'"defective":\s*\[\s*\]'
    count1 = len(re.findall(pattern1, text))
    # 匹配单独的："suspicious"
    pattern2 = r'"defective": \['
    count2 = len(re.findall(pattern2, text))
    return count1,count2

def main():
    path = "../results/large/RQ3/Ano-b/gpt-3.5-turbo/small_1"
    files = os.listdir(path)
    count_s = 0
    count_d = 0
    count = 0
    for file in files:
        file_path = f"{path}/{file}"
        text = new_main(file_path)
        cs1, cs2 = count_suspicious(text)
        cd1, cd2 = count_defective(text)
        print(file)
        print(f'匹配 "suspicious": [] 的数量：{cs1}')
        print(f'匹配 "suspicious" 的数量：{cs2}')
        print(f'匹配 "defective": [] 的数量：{cd1}')
        print(f'匹配 "defective" 的数量：{cd2}')
        if cs1 < cs2:
            count_s += 1
        if cd1 < cd2:
            count_d += 1
        if cs1 < cs2 or cd1 < cd2:
            count+=1
        print("="*100)
    print("suspicious:",count_s)
    print("defective:", count_d)
    print("count",count)

# def move(file,old_path,new_path):
#     file_path = f'{old_path}/{file}'
#     data = load_json_data(file_path)

def clean_df(file,old_path,new_path):
    file_path = f'{old_path}/{file}'
    data = load_json_data(file_path)
    llm_data = data[-1]['llm_result']
    if isinstance(llm_data, str):
        if "\"is_defect_commit\": \"YES\"" in llm_data:
            llm_data = {
                "is_defect_commit": "YES",
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
    # json_str = json.dumps(llm_data["defect_lines"], indent=4, ensure_ascii=False)
    # cd1, cd2 = count_defective(json_str)
    # cs1, cs2 = count_suspicious(json_str)
    old_answer = llm_data['is_defect_commit']
    # if old_answer != 'YES':
    #     llm_data['is_defect_commit'] = "YES" if ((cd1 < cd2) or (cs1 < cs2)) else "NO"
    new_answer = llm_data['is_defect_commit']
    new_file_path = f'{new_path}/{file}'
    with open(new_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"清洗完成:'{old_answer}'->'{new_answer}'")
    return new_answer

def clean():
    old_dir = '../results/large/RQ2/gpt-4o-mini-or'
    new_dir = '../results/large/RQ2/gpt-4o-mini'
    path = f"{old_dir}/small_20"
    count = 0
    with open('rq1_1730.txt', 'r', encoding='utf-8') as f:
        rq_list = [line.strip() for line in f.readlines()]
    for file in os.listdir(path):
        print(file)
        commit = file.split('.')[0]
        if commit not in rq_list:
            continue
        pred = 0
        for i in range(5):
            old_path = f"{old_dir}/small_{i + 20}"
            new_path = f"{new_dir}/small_{i + 1}"
            directory = Path(new_path)
            directory.mkdir(parents=True, exist_ok=True)
            answer = clean_df(file,old_path,new_path)
            pred += 1 if answer=='YES' else 0
            print("-"*100)
        print("="*100)
        count += 1 if pred >=3 else 0
    print("预测为缺陷个数:",count)



clean()
# with open('rq1_1730.txt', 'r', encoding='utf-8') as f:
#     rq1_list = [line.strip() for line in f.readlines()]
# with open('rq2_1000.txt', 'r', encoding='utf-8') as f:
#     rq2_list = [line.strip() for line in f.readlines()]
#
# difference = [item for item in rq1_list if item not in rq2_list]
#
# # 写入 extend.txt，每个元素占一行
# with open('extend.txt', 'w', encoding='utf-8') as f:
#     for item in difference:
#         f.write(item + '\n')








# main()


# data = [{'a': 1}]
# last_item = data[-1]
# last_item['a'] = 99
# print(data)
