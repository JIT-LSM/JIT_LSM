import json
import os
import shutil

# 提取rq1的提交
with open('../../rq1_1730.txt','r',encoding='utf-8') as f:
        rq1_commits = [line.strip() for line in f.readlines()]

data1_file = 'balanced_dataset_1000'
data2_file = 'extend_1039'
data3_file = 'unbalanced_dataset_1730'
# new_file = Path('unbalanced_dataset_1730')
# new_file.mkdir(parents=True,exists_ok=True)


projects = os.listdir(data1_file)


count1 = 0
count2 = 0
for project in projects:
    path1 = f'{data1_file}/{project}'
    path2 = f'{data2_file}/{project}'
    path3 = f'{data3_file}/{project}'
    commit1 = os.listdir(path1)
    commit2 = os.listdir(path2)
    for commit in commit1:
        if commit not in rq1_commits:
            continue
        count1 += 1
        source_folder = f'{path1}/{commit}'
        destination_folder = f'{path3}/{commit}'
        shutil.copytree(source_folder,destination_folder)
        print(f"文件夹 '{source_folder}' 已复制到 '{destination_folder}'")
    for commit in commit2:
        if commit not in rq1_commits:
            continue
        count2 += 1
        source_folder = f'{path2}/{commit}'
        destination_folder = f'{path3}/{commit}'
        shutil.copytree(source_folder, destination_folder)
        print(f"文件夹 '{source_folder}' 已复制到 '{destination_folder}'")
print(f"从'{data2_file}'中复制了{count1}个提交")
print(f"从'{data1_file}'中复制了{count2}个提交")

json_file = "unbalanced_dataset_1730.json"
json_struct = {}
for project in projects:
    json_struct[project] = {}
    path=f'{data3_file}/{project}'
    commits=os.listdir(path)
    for commit in commits:
        commit_data_file = f'{path}/{commit}/commit_data.json'
        with open(commit_data_file,'r',encoding='utf-8') as f:
            commit_data = json.load(f)
        json_struct[project][commit] = commit_data

with open(json_file,'w',encoding='utf-8') as f:
    json.dump(json_struct,f,indent=4)
print("json文件已经写入")