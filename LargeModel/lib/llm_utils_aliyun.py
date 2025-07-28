import time
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

# aiyun 接口
client = OpenAI(
    api_key="your-key"
)


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
