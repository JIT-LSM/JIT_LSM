from LargeModel.lib.AssistanceLLM import AssistantLLM as ALLM
from LargeModel.lib.Assistance_gemini import AssistantLLM as ALLM_ge
ALLM_MAP = {
        "gemini":ALLM_ge,
        "gpt":ALLM,
        "qwen":ALLM,
        "deepseek":ALLM_ge,
    }

class MainInterface():

    def __init__(self,project, commit, model, commit_data, data_dir,allow_checker,allow_inform):
        self._project = project
        self._commit = commit
        self._model = model
        self._data = commit_data
        self._data_dir = data_dir
        self._allow_checker = allow_checker
        self._allow_inform = allow_inform

    def get_commit_git_diff(self):
        commit_info = self._data["commit_info"]
        function_response = "The raw information obtained below through the git show command has not been processed in any way:\n"+commit_info
        print("[FC]","MianLLM call get_commit_git_diff")
        # print(function_response)
        return function_response

    def get_added_line_in_path(self,path):
        path_data = self._data[path]
        add_lines = path_data['added_lines']
        function_response = f"In {path} had added lines, please note that these lines do not have any positional relationship with each other; they are extracted from the overall code :\n"+'\n'.join(add_lines)
        print("[FC]",f"MainLLM call get_added_line_in_path ({path})")
        # print(function_response)
        return function_response

    def check_file_by_path(self, path, requirements):
        print("[FC]",f"MainLLM call check_file_by_path ({path})({requirements})")
        AssLLM = ALLM_MAP[self._model.split('-')[0]]
        assistant = AssLLM(self._project, self._commit, path, self._data[path], self._model, self._data_dir, requirements,self._allow_checker,self._allow_inform)
        function_response, message_list, assistant_log = assistant.run()
        return function_response, message_list, assistant_log

    # def get_function_by_name(self):
    #     return

    @property
    def funcmap(self):
        fmap = {
            "get_commit_git_diff": self.get_commit_git_diff,
            "get_added_line_in_path": self.get_added_line_in_path,
            "check_file_by_path": self.check_file_by_path
        }
        return fmap

