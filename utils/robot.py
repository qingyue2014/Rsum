from openai import OpenAI
import openai
import backoff
import time, tiktoken
import httpx

client = OpenAI(
    api_key="xxxxxx",
)
def gpt_response_results(args, prompt, model_name):
    for i in range(10):
        try:
            completion = client.chat.completions.create(
                model= args.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                    ],
                temperature=0,
                )
            break
        except:
            time.sleep(2)
    line_m = completion.choices[0].message.content.strip()
    return line_m


def gpt_memory_results(prompt, model_name):
    try:
        completion = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                    ],
            temperature=0,
            )
    except:
            time.sleep(5)
    line_m = completion.choices[0].message.content.strip()
    if "Updated memory:" in line_m:
        line_m = line_m.split("Updated memory:")[-1]
    return line_m
