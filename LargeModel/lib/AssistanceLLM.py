import json
import time

from LargeModel.lib.AssistanceLLM_interface import PathInterface
from LargeModel.lib.llm_utils import get_LLM_response as g_LLM_response
from LargeModel.lib.llm_utils_aliyun import get_LLM_response as g_LLM_res_aliyun


FOUNCTION_FILE = 'LargeModel/prompts/function_description.json'
SYSTEM_FILE = ''
class AssistantLLM():
    def __init__(self,project_name, commit_hash, path, path_data, model, data_dir, requirements,allow_checker=False, allow_inform=False):
        self._project = project_name
        self._commit = commit_hash
        self._data = path_data
        self._model = model
        self._requirements = requirements
        self._fc = PathInterface(path_data, project_name, commit_hash, path, data_dir)
        self._allow_checker = allow_checker
        self._allow_inform = allow_inform
        self._log = []

    def _append_to_messages(self,message):
        self.messages.append(message)

    def _pop_messages(self):
        self.messages.pop()

    def _record(self, type, message):
        self._log.append({
            "type":type,
            "message":message
        })

    @property
    def _system_message(self):
        system_message = """You are a meticulous code reviewer. 
For the given Java code file, your task is to carefully examine and analyze the newly added lines of code to determine if they will introduce any issues. 
More code information can be obtained through function calls. Please actively call these functions to get a more comprehensive code context to assist in your analysis.
And You can call the function a total of 15 times.
Please note that not all newly added code will introduce defects. It is possible that all the code in this submission file is secure. Also, please do not be overly cautious; simply judge based on the information you have obtained.
When you suspect a line of code has a defect, it's best to have a reasonable justification.
New let's check the file step by step.
"""
        return system_message

    @property
    def _function_descriptions(self):
        with open(FOUNCTION_FILE) as f:
            function_descriptions = json.load(f)
        return function_descriptions

    def get_LLM_response(self, **kwargs):
        response = g_LLM_response(**kwargs)
        return response

    def call_function(self, name, **args):
        func = self._fc.funcmap[name]
        function_response = func(**args)
        return function_response

    def initial(self):
        self.messages = []
        self._append_to_messages(
            {
                'role':'system',
                'content':self._system_message
            }
        )
        initial_prompt = f"""This file involves the addition of 2 lines of code. Please inspect these added lines.
The requirements for conducting the inspection are:{self._requirements}
Now, let's review the added code lines in this commit step by step, starting with the call to the function `get_commit_git_diff`."""
        self._append_to_messages({
            'role':'user',
            'content':initial_prompt,
        })
        self._append_to_messages({
            "role": "assistant",
            "content": "Null",
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

    def getTime(self):
        # 获取当前时间的时间戳
        timestamp = time.time()

        # 将时间戳转换为本地时间
        local_time = time.localtime(timestamp)

        # 格式化输出当前时间
        formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
        return formatted_time

    def step(self):
        response = self.get_LLM_response(
            model=self._model,
            messages=self.messages,
            functions=self._function_descriptions
        )
        if not response:
            self._append_to_messages({
                "content": "llm return null to jit-lsm.",
                "role": "assistant"
            })
            print("[error]", "LLM response NULL(By step)")
            self._record("Response Error", "Step response is none")
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
                function_response = self.call_function(function_name, **function_arguments)
            except Exception as e:
                self._record("Function Arguments Error", "Arguments Error")
                function_response = "Since the arguments not a json, you may call function in a wrong way, check and try again please!"
                print("[error]", "Function call error(By step)")
                print("[Traceback]", e)
            function_message = {
                "role": "function",
                "name": function_name,
                'content': function_response
            }
            self._append_to_messages(function_message)
            return False
        return True

    def check_need_call(self, message):
        print("[log]", "Checking whether need call a function")
        prompt = f"""Based on the context below, please determine whether there is an intention to call a function or whether there is a plan for next step. If yes, return ‘Yes’; otherwise, return ‘No’.
        {message}
        Any extra words are not needed. Please do not analyze or explain."""
        ms = [{
            'role': 'user',
            'content': prompt
        }]
        response = g_LLM_res_aliyun(
            model='qwen-turbo',
            messages=ms
        )
        content = response['choices'][0]['message']['content']
        print("[log]", "Checker result:", content)
        if 'Yes' in content:
            # 如果允许告知，则加入消息队列，反之则将前一条回复删除
            if self._allow_inform:
                self._append_to_messages({
                    'role': 'user',
                    'content': "Based on your response, it seems that you still need to call a function. Please call the function in the correct manner.",
                })
            else:
                self._pop_messages()
            return True
        return False

    def conclude(self):
        conclude_prompt = """Please provide a summary based on all your previous analyses. The focus of the summary should be on which line or lines may or will introduce defects, along with your reasoning for suspicion. An example of return format is as follows:
{
    "Summary":"Line 'values.put(actualName, actualValue);' may introduce a defect because... ; line 'getProject().setProperty(\\"ivy.revision\\", md.getModuleRevisionId().getRevision());' may introduce a defect because... ",
    "Further Check":"The function 'trim' is not implemented in the current file and appears to be externally imported. This could potentially involve a defect, so further investigation is needed."
}
All the content in your summary must be based on your previous analysis.
When you suspect or confirm that a line of code has a defect, it's best to have a reasonable justification."""
        self._append_to_messages({
            'role': 'user',
            'content': conclude_prompt
        })
        response = self.get_LLM_response(
            model=self._model,
            messages=self.messages)
        if not response:
            self._record("Response Error", "Conclusion response is none")
            print("[error]", 'LLM return None(By conclude)')
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

    def run(self):
        start_time = self.getTime()
        self.initial()
        i = 0
        last_check_need = False
        retry_calling = 0
        retry_count = 0
        while i < 15:
            print('.' * 200)
            self._record("Running Log", f"Step{i + 1}, it is the {retry_calling + 1} times of this step")
            print("[info]",f"Assistance Step{i + 1}, it is the {retry_calling + 1} times of this step")
            done = self.step()
            i += 1
            if done:
                if not self._allow_checker:
                    break
                if last_check_need:
                    if retry_calling >= 3:
                        retry_count += (retry_calling + 1)
                        self._record("Function Calling Error",
                                     "LLM attempted to invoke a function, but after three failed retries to call it in the correct format, the operation was forcibly terminated.")
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
        conclusion = self.conclude()
        self._record("Step Record", f"Totally {i + 1} Steps")
        self._record("Count Record", f"Function call fault {retry_count} times")
        end_time = self.getTime()
        self._record("Time Record", f"{start_time} TO {end_time}")
        return conclusion, self.messages, self._log





