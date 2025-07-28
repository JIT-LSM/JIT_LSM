import json
import os

from LargeModel.functions import spilt_code, hand_code_tags

def write_snippet_to_json(project, commit, path_seq, snippets, version,output_file):
    snippet_path_all = output_file+'/'+project+'/'+commit+'/path_'+str(path_seq)+'/'+version+'_snippet.json'
    snippet_path_dir = os.path.dirname(snippet_path_all)
    if not os.path.exists(snippet_path_dir):
        os.makedirs(snippet_path_dir)
    with open(snippet_path_all,'w',encoding='utf-8') as f:
        json.dump(snippets, f, ensure_ascii=False, indent=4)

def write_path2seq_to_json(project, commit,path2seq,output_file):
    path_all = output_file+'/'+project+'/'+commit+'/path2seq.json'
    path_dir = os.path.dirname(path_all)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    with open(path_all,'w',encoding='utf-8') as f:
        json.dump(path2seq, f, ensure_ascii=False, indent=4)

def find_cover_lines(start, end, map):
    cover_lines = {}
    for key in map.keys():
        if start <= map[key] <= end:
            cover_lines[key] = map[key]-1
    return cover_lines

def find_all_lines(map):
    all_lines = []
    for key in map.keys():
        all_lines.append(key)
    return all_lines

def cut_code2snippet_map(seq_map, method_map, code):
    code_lines = code.splitlines()
    snippets = []
    cover = set()
    for fragment in method_map:
        start = fragment['start_line']
        end = fragment['end_line']
        cover_lines = find_cover_lines(start, end, seq_map)
        cover.update(find_all_lines(cover_lines))
        signature = fragment['signature']
        snippets.append({
            "cover_lines":cover_lines,
            'signature': signature,
            "start":start-1,
            "end":end-1
        })
    all = set(find_all_lines(seq_map))
    uncover = all - cover
    for line in uncover:
        if line in cover:
            continue
        seq = seq_map[line]-1
        start = seq - 5 if seq-5 > 1 else 1
        end = seq + 6 if seq+6 <= len(code_lines) else len(code_lines)
        cover_lines = find_cover_lines(start, end, seq_map)
        cover.update(find_all_lines(cover_lines))
        snippets.append({
            "cover_lines": cover_lines,
            'signature':"",
            "start": start-1,
            "end": end-1
        })
    return snippets

def mark_code_after(start,end,map,code):
    code_lines = code.split('\n')
    marklines = []
    result_code = []
    for key in map.keys():
        if start <= map[key] <= end:
            marklines.append(map[key])
    for i in range(len(code_lines)):
        if i< start:
            continue
        if i == end+1:
            break
        if i in marklines:
            result_code.append(code_lines[i]+'  //New Add Line')
        else:
            result_code.append(code_lines[i])
    return '\n'.join(result_code)

def mark_code_before(start, end, seq, code):
    code_lines = code.split('\n')
    result_code = []
    for i in range(len(code_lines)):
        if i < start :
            continue
        if i == end + 1:
            break
        result_code.append(code_lines[i])
        if i == seq - 1:
            result_code.append('//A New Code Line Add in Here')
    return '\n'.join(result_code)

def after_code(after, seq_map):
    try:
        method_map = spilt_code.extract_methods_from_java_code(after)
        after_snippets = cut_code2snippet_map(seq_map, method_map, after)
    except:
        after_snippets = cut_code2snippet_map(seq_map, {},after)
    snippets = []
    for snippet in after_snippets:
        cover_lines = snippet['cover_lines']
        cover = find_all_lines(cover_lines)
        start = snippet['start']
        end = snippet['end']
        signature = snippet['signature']
        code = mark_code_after(start, end, cover_lines,after)
        snippets.append({
            "start":start,
            "end":end,
            "cover_lines":cover,
            "code":code,
            "signature":signature
        })
    return snippets

def before_code(before, seq_map):
    try:
        method_map = spilt_code.extract_methods_from_java_code(before)
        before_snippets = cut_code2snippet_map(seq_map, method_map, before)
    except:
        before_snippets = cut_code2snippet_map(seq_map, {}, before)
    snippets = []
    before_code_lines = before.split('\n')
    for snippet in before_snippets:
        cover_lines = snippet['cover_lines']
        cover = find_all_lines(cover_lines)
        start = snippet['start']
        end = snippet['end']
        code = '\n'.join(before_code_lines[start:end+1])
        signature = snippet['signature']
        snippets.append({
            "start": start,
            "end": end,
            "cover_lines": cover,
            "code": code,
            "signature": signature
        })
    return snippets

def cut(input_file,output_file):
    with open(input_file,'r',encoding='utf-8') as f:
        data = json.load(f)
    count = 0
    for project in data.keys():
        if project == 'commit_count':
            continue
        project_data = data[project]
        print(project)
        for commit in project_data.keys():
            commit_data = project_data[commit]
            print(commit)
            index = 0
            path2seq = {}
            next = False
            for path in commit_data['paths']:
                print(path)
                index += 1
                path2seq[path] = index
                path_data = commit_data[path]
                path_code = path_data['code']
                if path_code == '//Failed to fetch file content.':
                    print(commit)
                    print(path)
                    print("//Failed to fetch file content.")
                    next = True
                    break
                code_lines = path_code.split('\n')
                no_symbols, before, after, added, deleted = hand_code_tags.process_diff(code_lines)
                after_snippets = after_code(after,added)
                print('after done')
                before_snippets = before_code(before, deleted)
                print('before done')
                if next:
                    break
                write_snippet_to_json(project,commit,index,after_snippets,'after',output_file)
                write_snippet_to_json(project,commit,index,before_snippets,'before',output_file)
                path2seq[path] = index
            if next:
                continue
            write_path2seq_to_json(project,commit,path2seq,output_file)
            count += 1
    print(count)




