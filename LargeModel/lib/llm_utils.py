import time
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

# 清风阁
client = OpenAI(
    api_key='sk-sNIt2eRVMd2cMB6iNgLVoPhfzBucTiG6u2yoTFZROoiHLzr7',
    base_url='https://api.996444.cn/v1'
)

# # xiaoai代理 3接口
# client = OpenAI(
#     api_key = 'sk-7APVvLPdxJjh1VicbutuCBcKqQvXryd5yLlaA1FNfrxUDaPm',
#     base_url="https://xiaoai.plus/v1",
# )
# client = OpenAI(
#     api_key = 'sk-JdrNm63Hl4r86sZVgIwgWeeOAJxWwbTJBrGVoYwlWmQ21lRJ',
#     base_url="https://xiaoai.plus/v1",
# )

#  # 1号数据代理
# client = OpenAI(
#     api_key='sk-Owncle51YL03FLE7yr9PDGFd399prmJQORwDFdyr6kPzcT2H',
#     base_url="http://chatapi.littlewheat.com/v1",
# )


@retry(stop= stop_after_attempt(5),wait=wait_fixed(2))
def get_LLM_response(**kwargs):
    for _ in range(5):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.model_dump()  # If the above succeeds, we return here
        except Exception as e:
            save_err = e
            print(save_err)
            if "The server had an error processing your request." in str(e):
                time.sleep(1)
            else:
                break
        raise save_err


def check_quota():
    import http.client

    conn = http.client.HTTPSConnection("")
    payload = ''
    headers = {
        'Rix-Api-User': '100276'
    }
    conn.request("GET", "/api/token/key/sk-lEp6Wf3Uyrfn9I3tYBBpPLHYKZJI3K2gCpDz7spXOs4t1KH5", payload, headers)
    res = conn.getresponse()
    data = res.read()
    print(data.decode("utf-8"))

def say_hello():
    models_response = client.models.list()
    print("Available models:")
    for model in models_response.data:
        print(model.id)
    response = get_LLM_response(
        model='gemini-2.0-flash',
        messages=[{
            'role':'user',
            'content':'hello'
        }]
    )
    print(response['choices'][0]['message'])


# def test():
#     models_response = client.models.list()
#     # 打印模型 ID
#     print("Available models:")
#     for model in models_response.data:
#         print(model.id)

# test()
# check_quota()
# say_hello()

