import json

def write_commit_data(project,commit,dir,commit_data):
    with open(f"{dir}/{project}/{commit}/commit_data.json",'w',encoding='utf-8') as f:
        json.dump(commit_data,f,indent=4)

def cut(json_file, output_dir):
    with open(json_file,'r',encoding='utf-8') as f:
        data = json.load(f)
    for project in data.keys():
        if project == 'commit_count':
            continue
        project_data = data[project]
        print(project)
        for commit in project_data.keys():
            commit_data = project_data[commit]
            print(commit)
            write_commit_data(project,commit,output_dir,commit_data)


