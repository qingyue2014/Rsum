import openai
import backoff
import os
import re, time
from openai import OpenAI
api_key = "xxxxxx"
client = OpenAI(api_key="xxxxxx")
def gpt_summary_results(args, prompt):
    for i in range(100):
        try:
            completion = client.chat.completions.create(
                model= args.model_name,
                messages=[
            {"role": "system",
             "content": "You are an advanced AI language model with the ability to keep track of dialog information between speakers."},
            {"role": "user",
             "content": prompt,
             }],
                     temperature=0,
                )
            break
        except:
            time.sleep(5)
    line_m = completion.choices[0].message.content.strip()
    if 'Updated memory:' in line_m:
        line_m = line_m.split("Updated memory:")[-1]
    if '[Updated Memory]' in line_m:
        line_m = line_m.split("[Updated Memory]")[-1]
    return line_m

def gpt_response_results(args, prompt):
    for i in range(100):
        try:
            completion = client.chat.completions.create(
            model= args.model_name,
                    messages=[
            {"role": "system",
            "content": "You are an advanced AI language model designed to engage in personality-based conversations."},
        {"role": "user",
            "content": prompt,
            }],
                    temperature=0,
            )
            break
        except:
            time.sleep(5)
    line_m = completion.choices[0].message.content.strip()

    return line_m

def davinci_response_results(args, prompt):
    openai.api_key = api_key
    while True:
        try:
            response = openai.Completion.create(
           model="text-davinci-003",
        prompt=prompt,
        temperature=args.resp_temp,
        max_tokens=50,
            )
            break
        except:
            print("Error. Try Again!")
    message = response.choices[0].text
    message = message.split("User:")[0]
    return message

def davinci_summary_results(args, prompt):
    openai.api_key = api_key
    while True:
        try:
            response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=args.summ_temp,
            max_tokens=500,
            )
            break
        except:
            print("Error. Try Again!")

    message = response.choices[0].text
    return message


if __name__ == "__main__":
    prompt = "You are an advanced AI language model designed to engage in personality-based conversations. I need you respond to the user based on the provided dialogue context. Remember the response is natural and conversational, while staying within the word limit of 100 words. Can you write a prompt to finish the above task?"
    output = davinci_response_results(prompt, model_name='text-davinci-003')
    print(output)
