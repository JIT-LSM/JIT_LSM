import json
import os.path
import time
from LargeModel.lib.llm_utils import get_LLM_response as g_LLM_res
from LargeModel.lib.llm_utils_aliyun import get_LLM_response as g_LLM_res_aliyun
from LargeModel.lib.MainLLM_interface import MainInterface as MainLLM_Full
from LargeModel.lib.MainLLM_interface_no_assistance import MainInterface as MainLLM_only

FUNCTION_FILE = 'LargeModel/prompts/function.json'
FUNCTION_FILE_TOOL_CALL = 'LargeModel/prompts/function_ds.json'
SYSTEM_FILE = 'LargeModel/prompts/system_message_gemini.txt'
ONLY = 'LargeModel/prompts/only.json'


class MainLLM():
    def __init__(self, project_name, commit_hash, commit_data, model, data_dir, allow_tools=False, allow_assistance = False,allow_checker=False, allow_inform=False):
        self._project = project_name
        self._commit = commit_hash
        self._data = commit_data
        self._model = model
        self._allow_tools = allow_tools
        self._allow_assistance = allow_assistance
        self._allow_checker = allow_checker
        self._allow_inform = allow_inform
        self._path_num = 0
        self._record_message = []
        self._log = []
        self._fc = MainLLM_Full(project_name, commit_hash, model, commit_data,data_dir,allow_checker,allow_inform) \
            if allow_assistance \
            else \
            MainLLM_only(project_name, commit_hash, model, commit_data,data_dir,allow_checker,allow_inform)

    def _append_to_messages(self,message):
        self.messages.append(message)
        self._record_message.append(message)

    def _pop_messages(self):
        self.messages.pop()
        self._record_message.pop()

    def _record(self, type, message):
        self._log.append({
            "type": type,
            "message": message
        })

    def _print_log(self):
        print("[LOGS]")
        for log_record in self._log:
            print(log_record)

    @property
    def _system_message(self):
        with open(SYSTEM_FILE) as f:
            system_message = f.read().strip()
        return system_message

    @property
    def _function_descriptions(self):
        if self._allow_assistance:
            if self._model.split('-')[0] == 'gemini':
                function_description_file = FUNCTION_FILE_TOOL_CALL
            else:
                function_description_file = FUNCTION_FILE
            with open(function_description_file) as f:
                function_descriptions = json.load(f)
        else:
            with open(ONLY) as f:
                function_descriptions = json.load(f)
        return function_descriptions

    def get_LLM_response(self, **kwargs):
        return g_LLM_res(**kwargs)

    def write_message_to_json(self, round,seq,output_dir):
        result_path = f'{output_dir}/{self._model}'
        full_path = f'{result_path}/full_{round}'
        small_path = f'{result_path}/small_{round}'
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        if not os.path.exists(small_path):
            os.makedirs(small_path)
        result = self.messages[-1]['content']
        if self._allow_tools:
            conclusion = self.messages[-3]['content']
        else:
            conclusion = ""
        with open(f'{small_path}/{self._project}_{seq}.json', 'w', encoding='utf-8') as fe:
            try:
                result_json = json.loads(result)
            except:
                if not result:
                    result_json = json.loads("{}")
                else:
                    result_json = result
            json_structure = []
            for path in self._data['paths']:
                json_structure.append({
                    "path": path,
                    "add_lines": self._data[path]['added_lines'],
                    "add_buggy": self._data[path]['added_buggy']
                })
            json_structure.append({
                "llm_result": result_json,
                "llm_conclusion": conclusion
            })
            json.dump(json_structure, fe, ensure_ascii=False, indent=4)

        self._append_to_messages({
            "log_record": self._log
        })
        with open(f'{full_path}/{self._project}_{seq}.json','w',encoding='utf-8') as f:
            json.dump(self._record_message, f, indent=4)


    def check_need_call(self, message):
        print("[LOG]","Checking whether need call a function")
        prompt = f"""Based on the text below, please determine whether there is a need to call a function, or any follow-up actions or plans that need to be carried out. If yes, return ‘Yes’; otherwise, return ‘No’.
{message}
Any extra words are not needed. Please do not analyze or explain."""
        ms = [{
            'role':'user',
            'content':prompt
        }]
        response = g_LLM_res_aliyun(
            model='qwen-turbo',
            messages=ms
        )
        content = response['choices'][0]['message']['content']
        print("[LOG]","Checker result:",content)
        if 'Yes' in content:
            #如果允许告知，则加入消息队列，反之则将前一条回复删除
            if self._allow_inform:
                self._append_to_messages({
                    'role': 'user',
                    'content': "Based on your response, it seems that you still need to call a function. Please call the function in the correct manner.",
                })
            else:
                self._pop_messages()
            return True
        return False

    def call_function(self, name, **kwargs):
        func = self._fc.funcmap[name]
        if name == "check_file_by_path":
            function_response, message_list, assistant_log = func(**kwargs)
            self._record_message.append({
                "Assistance message": message_list,
                "Assistance log": assistant_log
            })
        else:
            function_response = func(**kwargs)
        return function_response

    def initial_NFC(self):
        self.messages = []
        Commit_message = "The raw information obtained below through the git show command has not been processed in any way:\n" + \
                         self._data['commit_info']
        paths = self._data['paths']
        self._path_num = len(paths)
        Change_hunk = "The files and corresponding newly added code lines involved in this commit are as follows (note that the code lines are discrete and do not have a sequential relationship):\n"
        for path in paths:
            code_line_str = '\n'.join(self._data[path]['added_lines'])
            Change_hunk += f"{path} :\n{code_line_str}\n"
        task = """You are a meticulous code review expert, highly skilled at reviewing code submitted through Git.
        Your current task is to check whether the newly added code will introduce defects based on the submitted information and the differential code snippets.
        Please note that not all commits will introduce defects; please judge carefully."""
        Output_Format = """I would like to see a summary of all the files. Return the code lines specifically as json types directly.
        Below is an example of the return json format.
        { 
            "is_defect_commit": YES (Return 'YES' if you confirm that this commit will introduce defects or return 'NO'), 
            "defect_lines":[
                {
                    "path":"src/java/org/apache/ivy/Ivy14.java",
                    "suspicious":[
                        "getProject().setProperty(\\"ivy.revision\\", md.getModuleRevisionId().getRevision());",                 
                        "getProject().setProperty(\\"ivy.configurations\\", mergeConfs(md.getConfigurationsNames()));"
                     ],
                    "defective":[
                        "values.put(actualName, actualValue);"
                     ],
                    "explanation":"The code line 'getProject().setProperty(\\"ivy.configurations\\", mergeConfs(md.getConfigurationsNames()));' may cause an infinite loop because... ;The code line 'getProject().setProperty(\\"ivy.configurations\\", mergeConfs(md.getConfigurationsNames()));' will cause null pointer exception because..."
                },
                {
                    "path":"src/java/org/apache/ivy/core/settings/XmlSettingsParser.java",
                    "suspicious": [],
                    "defective": [
                        "Message.deprecated('defaultCache' is deprecated, use 'caches[@defaultCacheDir]' instead (' + settings + ')');"
                    ],
                    "explanation":"..."
                }
            ]
        }"""
        split_tag = "\n###\n"
        prompt = f"{task}{split_tag}{Commit_message}{split_tag}{Change_hunk}{split_tag}{Output_Format}"
        self._append_to_messages({
            "role": "user",
            "content": prompt
        })

    def initial(self):
        self.messages = []
        path_in_commit = []
        path_code_only = []
        self._append_to_messages(
            {
                'role':'system',
                'content':self._system_message
            }
        )
        for path in self._data['paths']:
            added_lines = self._data[path]['added_lines']
            path_in_commit.append(path)
            if added_lines:
                path_code_only.append(path)
        self._path_num = len(path_code_only)
        path_in_commit_str = '\n'.join(path_in_commit)
        path_code_only_str = '\n'.join(path_code_only)
        initial_prompt = f"""For the project '{self._project}', its commit '{self._commit}', involves the following files:
{path_in_commit_str}

Among the files, those involving changes to code lines include:
{path_code_only_str}

Now, let's review the added code lines in this commit step by step, starting with the call to the function `getCommitGitDiff`."""
        self._append_to_messages({
            'role':'user',
            'content':initial_prompt,
        })
        self._append_to_messages({
            "role": "assistant",
            "content": "Step 1: obtain comprehensive information about this commit. Call function 'get_commit_git_diff'.",
            "function_call": {
                "name": "get_commit_git_diff",
                "arguments": "{}"
            }
        })
        self._append_to_messages({
            "role": "function",
            "name": "get_commit_git_diff",
            "content": json.dumps(self._fc.funcmap["get_commit_git_diff"]())
        })

    def initial_toolcall(self):
        self.messages = []
        path_in_commit = []
        path_code_only = []
        self._append_to_messages(
            {
                'role': 'system',
                'content': self._system_message
            }
        )
        for path in self._data['paths']:
            added_lines = self._data[path]['added_lines']
            path_in_commit.append(path)
            if added_lines:
                path_code_only.append(path)
        self._path_num = len(path_code_only)
        path_in_commit_str = '\n'.join(path_in_commit)
        path_code_only_str = '\n'.join(path_code_only)
        initial_prompt = f"""For the project '{self._project}', its commit '{self._commit}', involves the following files:
{path_in_commit_str}

Among the files, those involving changes to code lines include:
{path_code_only_str}

Now, let's review the added code lines in this commit step by step, starting with the call to the function `getCommitGitDiff`."""
        self._append_to_messages({
            'role': 'user',
            'content': initial_prompt,
        })

    def step_toolcall(self):
        response = self.get_LLM_response(
            model=self._model,
            messages=self.messages,
            tools=self._function_descriptions
        )
        if not response:
            self._append_to_messages({
                "content": "llm return null to jit-lsm.",
                "role": "assistant"
            })
            print("[ERROR]","LLM response NULL(By step)")
            self._record("Response Error", "Step response is none")
            return True
        response_message = response['choices'][0]['message']
        if not response_message['content']:
            response_message['content'] = "null"
        self._append_to_messages(response_message)
        if response_message['tool_calls']:
            for i in range(len(response_message['tool_calls'])):
                function_call_id = response_message['tool_calls'][i]['id']
                function_call = response_message['tool_calls'][i]['function']
                function_name = function_call['name']
                try:
                    function_arguments =  json.loads(function_call['arguments'])
                    print("[DBUG]", "Arguments Right")
                    function_response = self.call_function(function_name, **function_arguments)
                except Exception as e:
                    function_response = "You may call function in a wrong way, check and try again please!"
                    print(f"step ERROR: call function fault {e}")
                function_message = {
                    "role": "tool",
                    "tool_call_id": function_call_id,
                    'content': function_response
                }
                self._append_to_messages(function_message)
            return False
        return True

    def step(self):
        response = self.get_LLM_response(
            model=self._model,
            messages=self.messages,
            functions=self._function_descriptions
        )
        if not response:
            self._append_to_messages({
                "content":"llm return null to jit-lsm.",
                "role":"assistant"
            })
            print("[ERROR]","LLM response NULL(By step)")
            self._record("Response Error","Step response is none")
            return True
        response_message = response['choices'][0]['message']
        if 'tool_calls' in response_message:
            del response_message['tool_calls']
        if not response_message['content']:
            response_message['content'] = "null"
        self._append_to_messages(response_message)
        if response_message['function_call']:
            function_call = response_message['function_call']
            function_name = function_call['name']
            try:
                function_arguments = json.loads(function_call['arguments'])
                print("[DBUG]","Arguments Right")
                function_response = self.call_function(function_name, **function_arguments)
            except Exception as e:
                self._record("Function Arguments Error","Arguments Error")
                function_response = "Since the arguments not a json, you may call function in a wrong way, check and try again please!"
                print("[ERROR]","Function call error(By step)")
                print("[Traceback]",e)
            function_message = {
                "role": "function",
                "name": function_name,
                'content': function_response
            }
            self._append_to_messages(function_message)
            return False
        return True

    def finish(self):
        conclude_prompt = f"""The results provided must be based on your previous analysis. Now let's make a conclusion. 
As the review is confirmed to be complete, please summarize and categorize the code lines identified during the analysis process into two categories: Suspicious Code Lines and Defective Code Lines.
Among the suspicious code lines are those that you suspect but cannot confirm will introduce defects; defective code lines refer to those in this change that you have confirmed will introduce defects.
This submission involves changes to {self._path_num} Java files. I would like to see a summary of all the files, and I expect your summary to be based on all the analysis you have conducted previously.
Is_defect_commit indicates whether this commit may introduce defects, Return YES or NO
Return the code lines specifically as json types directly. Below is an example of the return json format. If not available, return None without any explanation:""" + """\n{
"is_defect_commit": YES,
"defect_lines":
    [
        {
            "path":"src/java/org/apache/ivy/Ivy14.java",
            "suspicious":["getProject().setProperty(\\"ivy.revision\\", md.getModuleRevisionId().getRevision());",
                "getProject().setProperty(\\"ivy.configurations\\", mergeConfs(md.getConfigurationsNames()));"
            ],
            "defective":["values.put(actualName, actualValue);"]
            "explanation":"The code line 'getProject().setProperty(\\"ivy.configurations\\", mergeConfs(md.getConfigurationsNames()));' may cause an infinite loop because... ;The code line 'getProject().setProperty(\\"ivy.configurations\\", mergeConfs(md.getConfigurationsNames()));' will cause null pointer exception because..."
        },
        {
            "path": "src/java/org/apache/ivy/core/settings/XmlSettingsParser.java",
            "suspicious": [],
            "defective": ["Message.deprecated('defaultCache' is deprecated, use 'caches[@defaultCacheDir]' instead (' + settings + ')');"]
            "explanation":"..."
        }
    ]
}
"""
        self._append_to_messages(
            {
                "role": "user",
                "content": conclude_prompt
            }
        )
        response = self.get_LLM_response(
            model=self._model,
            messages=self.messages,
            response_format={ "type": "json_object" }
        )
        if not response:
            self._record("Response Error", "Finish response is none")
            print("[ERROR]",'LLM return None(By finish)')
            self._append_to_messages({
                "role": "assistant",
                "content": '{}'
            })
            return '{}'
        response_message = response['choices'][0]['message']
        if 'tool_calls' in response_message:
            del response_message['tool_calls']
        if not response_message['content']:
            response_message['content'] = ""
        self._append_to_messages(response_message)
        return response_message['content']

    def conclude(self):
        conclude_prompt = 'Since you are confident that the inspection has been completed, please provide a summary based on all your previous analyses. The focus of the summary should be on which line or lines may or will introduce defects, along with your reasoning for suspicion.'
        self._append_to_messages({
            'role': 'user',
            'content': conclude_prompt
        })
        response = self.get_LLM_response(
            model=self._model,
            messages=self.messages)
        if not response:
            self._record("Response Error", "Conclusion response is none")
            print("[ERROR]",'LLM return None(By conclude)')
            self._append_to_messages({
                "role": "assistant",
                "content": 'Null'
            })
            return ''
        response_message = response['choices'][0]['message']
        if 'tool_calls' in response_message:
            del response_message['tool_calls']
        if not response_message['content']:
            response_message['content'] = ""
        self._append_to_messages(response_message)
        return response_message['content']

    def getTime(self):
        # 获取当前时间的时间戳
        timestamp = time.time()

        # 将时间戳转换为本地时间
        local_time = time.localtime(timestamp)

        # 格式化输出当前时间
        formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
        return formatted_time

    def run(self):
        start_time = self.getTime()
        if self._allow_tools:
            if self._model.split('-')[0] == 'gemini':
                self.initial_toolcall()
            else:
                self.initial()
            i = 0
            last_check_need = False
            retry_calling = 0
            retry_count = 0
            while i < 15:
                print('-'*200)
                self._record("Running Log",f"Step{i+1}, it is the {retry_calling+1} times of this step")
                print("[LOG]",f"Step{i+1}, it is the {retry_calling+1} times of this step")
                if self._model.split('-')[0] == 'gemini':
                    done = self.step_toolcall()
                else:
                    done = self.step()
                i += 1
                if self._allow_checker:
                    if done:
                        if last_check_need:
                            if retry_calling >=3:
                                retry_count += (retry_calling+1)
                                self._record("Function Calling Error", "LLM attempted to invoke a function, but after three failed retries to call it in the correct format, the operation was forcibly terminated.")
                                break
                            self._pop_messages()
                            retry_calling += 1
                            i -= 1
                            time.sleep(2)
                            continue
                        need = self.check_need_call(self.messages[-1]['content'])
                        if need:
                            last_check_need = True
                            retry_calling += 1
                            i -= 1
                            time.sleep(2)
                            continue
                        else:
                            break
                    last_check_need = False
                    retry_count += retry_calling
                    retry_calling = 0
                else:
                    if done:
                        break
            conclusion = self.conclude()
            result = self.finish()
            self._record("Step Record",f"Totally {i+1} Steps")
            self._record("Count Record",f"Function call fault {retry_count} times")
            end_time = self.getTime()
            self._record("Time Record",f"{start_time} TO {end_time}")
            return result, conclusion
        else:
            self.initial_NFC()
            self._record("Running Log", "Step1, it is the 1 times of this step")
            response = self.get_LLM_response(
                model=self._model,
                messages=self.messages,
                response_format={"type": "json_object"}
            )
            if not response:
                self._append_to_messages({
                    "role": "assistant",
                    "content": '{}'
                })
                response_message = self.messages[-1]
                self._record("Response Error","Step response is none")
            else:
                response_message = response['choices'][0]['message']
            if not response_message['content']:
                response_message['content'] = "{}"
            self._append_to_messages(response_message)
            self._record("Step Record", "Totally 1 Steps")
            self._record("Count Record", f"Function call fault 0 times")
            end_time = self.getTime()
            self._record("Time Record", f"{start_time} TO {end_time}")
            return response_message['content'],""

