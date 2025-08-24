# -*- coding: utf-8 -*-
import sys 
import json, os
from openai import OpenAI
import copy,time
import openai
from utils import evaluation
import httpx
from tqdm import tqdm
from rag import retrieval_content

DEFINED_PROMPT="You are an advanced AI designed for engaging in a personality-based conversations. You will be provided with personal preferences and experiences of speakers (the assistant and the user), and a dialogue context. When responding, consider maintaining a conversational and fluent tone. Responses should be contextually relevant, consistent with given memory, aiming to keep the conversation flowing. Human queries are labeled 'User:', while your replies are marked 'Assistant:'. Your goal is to provide engaging and coherent responses based on the dialogue context. The response is in the form of text and cannot contain emoticons or special characters.The following is the case you need to test:\nThe personality is:{persona}\nThe memory is:{history}\nThe test dialogue context is:{dialog}\nSo the response to the user is: Assistant:"

client = OpenAI(
    api_key="xxxxx",
)

def gpt_response_results(prompt):
    for i in range(100):
        try:
            completion = client.chat.completions.create(
                model= "gpt-3.5-turbo-0301",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                    ],
                     temperature=0,
                )
            break
        except:
            time.sleep(2)
    #import pdb; pdb.set_trace()
    line_m = completion.choices[0].message.content.strip()
    return line_m

class LLMClientSimple:

    def __init__(self,gen_config=None):
        
        self.disable_tqdm = False
        self.gen_config = gen_config 

    def generate_text_simple(self,prompt,prompt_num,language='en'):
        self.gen_config['n'] = prompt_num
        retry_times,count = 100,0
        response = None
        while response is None and count<retry_times:
            try:
                request = copy.deepcopy(self.gen_config)
                # print(prompt)
                if language=='cn':
                    message = [
                    {"role": "system", "content": "以下是一个人类和一个聪明、懂心理学的AI助手之间的对话记录。"},
                    {"role": "user", "content": "你好！请帮我对对话内容归纳总结"},
                    {"role": "system", "content": "好的，我会尽力帮你的。"},
                    {"role": "user", "content": f"{prompt}"}]
                else:
                    message = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Please help me summarize the content of the conversation.{prompt}"}]

                response = client1.chat.completions.create(
                    **request, messages=message)
                # print(prompt)
            except Exception as e:
                print(e)
                if 'This model\'s maximum context' in str(e):
                        cut_length = 1800-200*(count)
                        print('max context length reached, cut to {}'.format(cut_length))
                        prompt = prompt[-cut_length:]
                        response=None
                count+=1
        if response:
            task_desc = response.choices[0].message.content.strip() #[response['choices'][i]['text'] for i in range(len(response['choices']))]
        else:
            task_desc = ''
        return task_desc
    

chatgpt_config = {"model": "gpt-3.5-turbo-0301",
        "temperature": 0,
        "stop": ["<|im_end|>", "¬人类¬"]
        }

llm_client = LLMClientSimple(chatgpt_config)

def summarize_content_prompt(content,user_name,boot_name,language='en'):
    prompt = '请总结以下的对话内容，尽可能精炼，提取对话的主题和关键信息。如果有多个关键事件，可以分点总结。对话内容：\n' if language=='cn' else 'Please summarize the following dialogue as concisely as possible, extracting the main themes and key information. If there are multiple key events, you may summarize them separately. Dialogue content:\n'
    prompt += content
    prompt += ('\n总结：' if language=='cn' else '\nSummarization：')
    return prompt

def summarize_overall_prompt(content,language='en'):
    prompt = '请高度概括以下的事件，尽可能精炼，概括并保留其中核心的关键信息。概括事件：\n' if language=='cn' else "Please provide a highly concise summary of the following event, capturing the essential key information as succinctly as possible. Summarize the event:\n"
    for date,summary_dict in content.items():
        summary = summary_dict['content']
        prompt += (f"\n时间{date}发生的事件为{summary.strip()}" if language=='cn' else f"At {date}, the events are {summary.strip()}")
    prompt += ('\n总结：' if language=='cn' else '\nSummarization：')
    return prompt

def summarize_overall_personality(content,language='en'):
    prompt = '以下是用户在多段对话中展现出来的人格特质和心情，以及当下合适的回复策略：\n' if language=='cn' else "The following are the user and assistant's exhibited personality traits throughout multiple dialogues:"
    for date,summary in content:
        prompt += (f"\n在时间{date}的分析为{summary.strip()}" if language=='cn' else f"At {date}, the analysis shows {summary.strip()}")
    #prompt += ('\n请总体概括用户的性格和AI恋人最合适的回复策略，尽量简洁精炼，高度概括。总结为：' if language=='cn' else "Please provide a highly concise and general summary of the user and assistant's personality and the most appropriate response strategy for the AI Assistant, summarized as:")
    return prompt

def summarize_person_prompt(content,user_name,boot_name,language):
    prompt = f'请根据以下的对话分别推测总结{user_name}和{boot_name}的性格特点和心情。对话内容：\n' if language=='cn' else f"Based on the following dialogue, please summarize {user_name}'s and {boot_name}'s personality traits. Dialogue content:\n"
    prompt += content
    #prompt += (f'\n{user_name}和{boot_name}的性格特点、心情、{boot_name}的回复策略为：' if language=='cn' else f"\n{user_name}'s and {boot_name}'s personality traits, emotions, and {boot_name}'s response strategy are:")
    return prompt


def run_memorybank(args, test_dataset, language='en'):
    bot_name = 'Assistant'
    user_name = 'User'
    gen_prompt_num = 1
    pred_dicts = {}
    all_preds, all_trues = [],[]

    for test_dial in tqdm(test_dataset):
        memory = {}
        history = test_dial['history']
        memory['summary'] = {}
        memory['personality'] = {}
        for idx, content in enumerate(history):
            hisprompt = summarize_content_prompt(content,user_name, bot_name,language)
            person_prompt = summarize_person_prompt(content,user_name, bot_name,language)

            his_summary = gpt_response_results(hisprompt)
            memory['summary'][idx] = {'content':his_summary}
            if args.dataset == 'msc':
                person_summary = llm_client.generate_text_simple(prompt=person_prompt,prompt_num=gen_prompt_num,language=language)
                memory['personality'][idx] = person_summary
        
        overall_his_prompt = summarize_overall_prompt(memory['summary'],language=language)
        overall_person_prompt = summarize_overall_personality(list(memory['personality'].items()),language=language)
        memory['overall_history'] = llm_client.generate_text_simple(prompt=overall_his_prompt,prompt_num=gen_prompt_num,language=language)
        if args.dataset == 'msc':
            memory['overall_personality'] = llm_client.generate_text_simple(prompt=overall_person_prompt,prompt_num=gen_prompt_num,language=language)
        
        if args.dataset == 'msc':
            DEFINED_PROMPT="You are an advanced AI designed for engaging in a personality-based conversations. You will be provided with personal preferences and experiences of speakers (the assistant and the user), and a dialogue context. When responding, consider maintaining a conversational and fluent tone. Responses should be contextually relevant, consistent with given memory, aiming to keep the conversation flowing. Human queries are labeled 'User:', while your replies are marked 'Assistant:'. Your goal is to provide engaging and coherent responses based on the dialogue context. The response is in the form of text and cannot contain emoticons or special characters.The following is the case you need to test:\nThe personality is:{persona}\nThe memory is:{history}\nThe test dialogue context is:{dialog}\nSo the response to the user is: Assistant:"
            input_str = DEFINED_PROMPT.format_map({"history": memory['overall_history'], "persona": memory['overall_personality'], "dialog": test_dial['window']})
        else:
            DEFINED_PROMPT="You are an advanced AI designed for engaging in a personality-based conversations. You will be provided with a history, including personal preferences and experiences of speakers (the assistant and the user), and a dialogue context. When responding, consider maintaining a conversational and fluent tone. Responses should be contextually relevant, consistent with given memory, aiming to keep the conversation flowing. Human queries are labeled 'User:', while your replies are marked 'Assistant:'. Your goal is to provide engaging and coherent responses based on the dialogue context. The response is in the form of text and cannot contain emoticons or special characters.The following is the case you need to test:\nThe history is:{history}\nThe test dialogue context is:{dialog}\nSo the response to the user is: Assistant:"
            input_str = DEFINED_PROMPT.format_map({"history": memory['overall_history'], "dialog": test_dial['window']})

        
        response = gpt_response_results(input_str)
        if args.dataset == 'msc':
            pred_dicts[test_dial['dial_id']] = {'prediction': response, 'label': test_dial['response'], 'memory': memory['overall_history'], 'persona':memory['overall_personality']}
        else:
            pred_dicts[test_dial['dial_id']] = {'prediction': response, 'label': test_dial['response'], 'memory': memory['overall_history']}
        all_preds.append(response)
        all_trues.append(test_dial['response'])

    wfile= os.path.join(args.saving_dir, f"{args.operation}_{args.mode}_sid{args.session_id}.json")
    with open(wfile,"w", encoding='utf-8') as f: 
        f.write(json.dumps(pred_dicts, ensure_ascii=False, indent=4))  

    results = evaluation.evaluate_corpus(all_preds, all_trues)
    result = {k: round(v * 100, 4) for k, v in results.items()}
    print(result)
    with open(f'{args.logger_file}', 'a') as f:
        f.write(str(args)+"\n")
        f.write(str(result)+"\n")

def run_memorybank_response(args, test_dataset):
    file_path = os.path.join("save_msc/memorybank", f"infer_full_sid{args.session_id}.json")
    with open(file_path, 'r') as f:
        memory = json.load(f)

    pred_dicts = {}
    all_preds = []
    all_trues = []
    for test_dial in tqdm(test_dataset):
        memory_dial = memory[test_dial['dial_id']]
        input_str = DEFINED_PROMPT.format_map({"persona": memory_dial['persona'], "dialog": test_dial['window']})
        response = gpt_response_results(input_str, args.model_name)
        pred_dicts[test_dial['dial_id']] = {'prediction': response, 'label': test_dial['response'], 'persona':memory_dial['persona']}
        all_preds.append(response)
        all_trues.append(test_dial['response'])
    
    wfile= os.path.join(args.saving_dir, f"{args.operation}_{args.mode}_sid{args.session_id}_new.json")
    with open(wfile,"w", encoding='utf-8') as f: 
        f.write(json.dumps(pred_dicts, ensure_ascii=False, indent=4))  

    results = evaluation.evaluate_corpus(all_preds, all_trues)
    result = {k: round(v * 100, 4) for k, v in results.items()}
    print(result)
    with open(f'{args.logger_file}', 'a') as f:
        f.write(str(args)+"\n")
        f.write(str(result)+"\n")
