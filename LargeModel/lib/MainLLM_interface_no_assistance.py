import json
class MainInterface():

    def __init__(self,project, commit, model, commit_data, DIR,allow_checker,allow_inform):
        self._project = project
        self._commit = commit
        self._model = model
        self._data = commit_data
        self._dir = DIR


    def path2seq(self,path):
        commit_path = self._dir + '/' + self._project + '/' + self._commit + '/path2seq.json'
        with open(commit_path, 'r', encoding='utf-8') as f:
            map = json.load(f)
        return map[path]

    def get_commit_git_diff(self):
        commit_info = self._data["commit_info"]
        function_response = "The raw information obtained below through the git show command has not been processed in any way:\n"+commit_info
        print("call get_commit_git_diff")
        return function_response

    def get_file_git_diff(self,path):
        print("call get_file_git_diff")
        commit_mess = self._data[path]['summary']
        function_reponse = "The raw information obtained below through the git show command has not been processed in any way:\n"+commit_mess
        return function_reponse

    def get_added_lines_in_path(self, path):
        print('call get_added_lines_in_path')
        #新增代码行直接读数据
        added_lines = self._data[path]['added_lines']
        added_lines_str = '\n'.join(added_lines)
        function_response = "In this file, the code lines below had been added, please note that these lines do not have any positional relationship with each other; they are extracted from the overall code. For the true contextual relationships, please call the 'get_added_line_context'.:\n"+added_lines_str
        # 给代码行分块
        path_seq = str(self.path2seq(path))
        path_path = self._dir + '/' + self._project + '/' + self._commit + '/' + 'path_' + path_seq + '/' + 'after_snippet.json'
        added_leave = set(added_lines)
        with open(path_path, 'r', encoding='utf-8') as f:
            snippets = json.load(f)
        for snippet in snippets:
            cover_lines = snippet['cover_lines']
            cover = set(cover_lines)
            added_covered = added_leave & cover
            if added_covered:
                added_leave = added_leave - cover
                added_covered_list = list(added_covered)
                sta = "\n\nThe following code lines share the same context snippet. When calling 'get_added_line_context', you only need to select any one of these lines:\n"
                added_covered_str = '\n'.join(added_covered_list)
                function_response += (sta + added_covered_str)
            if not added_leave:
                break
        return function_response


    def get_added_line_context(self, code_line, version, path):
        print("call get_added_line_context")
        path_seq = str(self.path2seq(path))
        if version != 'After' and version != 'Before':
            function_response = "No such version. Check and try again please."
            print(function_response)
            return function_response
        else:
            ver = version.lower()
        added_lines = self._data[path]['added_lines']
        added = set(added_lines)
        if code_line not in added_lines:
            function_response = "No such newly added code line. Check and try again please."
            print(function_response)
            return function_response
        path_path = self._dir + '/' + self._project + '/' + self._commit + '/' + 'path_' + path_seq + '/' + ver + '_snippet.json'
        with open(path_path, 'r', encoding='utf-8') as f:
            snippets = json.load(f)
        for snippet in snippets:
            cover_lines = snippet['cover_lines']
            cover = set(cover_lines)
            if code_line in cover_lines:
                covered = list(added & cover)
                function_response = "Below is the context we have found for the added lines of code. For the context after the changes, which may involve multiple new lines of code, we have marked all such lines with ‘//New Add Line’.\n" + snippet['code']
                # print(function_response)
                print('found')
                return function_response
        function_response = "Something wang, check and try again please."
        print(function_response)
        return function_response

    def get_function_code_by_name(self, function_name, version, path):
        print("call get_function_code_by_name")
        path_seq = str(self.path2seq(path))
        ags = f"name = {function_name}, version = {version}"
        added_lines = self._data[path]['added_lines']
        added = set(added_lines)
        if version != 'After' and version != 'Before':
            print("No such version. Check and try again please.")
            return "No such version. Check and try again please."
        else:
            ver = version.lower()
        path_path = self._dir + '/' + self._project + '/' + self._commit + '/' + 'path_' + path_seq + '/' + ver + '_snippet.json'
        with open(path_path, 'r', encoding='utf-8') as f:
            snippets = json.load(f)
        found = []
        count = 0
        for snippet in snippets:
            if function_name == snippet['signature'].split('(')[0]:
                cover_lines = snippet['cover_lines']
                cover = set(cover_lines)
                found.append(snippet["code"]+'\n')
                covered = list(added & cover)
                count+=1
        if found:
            found_str = '\n'.join(found)
            print("found")
            return f"There are {count} function name '{function_name}' were found:\n{found_str}"
        else:
            print("no found")
            return f"The function name {function_name} is not found in this file. If this function is crucial for the current inspection, please point it out. Then, temporarily skip this section and continue with the other inspection tasks."


    def get_function_by_name(self):
        return

    @property
    def funcmap(self):
        fmap = {
            "get_commit_git_diff": self.get_commit_git_diff,
            "get_file_git_diff":self.get_file_git_diff,
            "get_added_lines_in_path":self.get_added_lines_in_path,
            "get_function_code_by_name":self.get_function_code_by_name,
            "get_added_line_context":self.get_added_line_context
        }
        return fmap

