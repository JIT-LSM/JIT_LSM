import time
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed


client = OpenAI(
    api_key='your-key'
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

# test()
# check_quota()
# say_hello()

