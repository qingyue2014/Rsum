from openai import OpenAI
import os
import json
import re
import time
from tqdm import tqdm
import random
random.seed(42)

prompts = json.load(open("data/msc_dialogue/prompts.json", "r"))

client = OpenAI(
    api_key="xxxxxx",
)
def gpt_response_results(prompt, model_name):
    for _ in range(100):
        try:
            completion = client.chat.completions.create(
                model= model_name,
            messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                    ],
                     temperature=0,
                )
            break
        except:
            time.sleep(5)
    line_m = completion.choices[0].message.content.strip()
    return line_m


def run_llm_win(args, test_dataset):
    pre_defined = ['Response 1 is better', 'Response 1 is slightly better', 
                   'Response 2 is better', 'Response 2 is slightly better']
    results = [0,0,0,0]
    model1, model2 = 'gpt-3.5-old', 'memochat'
    
    eval_name = os.path.join(args.saving_dir, f"{model1}_{model2}_win_rate.json")
    evaluation = json.load(open(eval_name,"r"))
    for eval in tqdm(evaluation):
        judge_result = eval['evaluation']
        flag = 1
        input_str = f"There is a paragraph: {judge_result}. You should give the final conclusion. Your conclusion must be given from the four choices:{pre_defined}"
        new_result = gpt_response_results(input_str, 'gpt-3.5-turbo-1106')
        for pre_idx, pre in enumerate(pre_defined):
            if flag == 1:
                if pre in new_result:
                    results[pre_idx] = results[pre_idx] + 1
                    flag = 0
                    break

        if flag == 1:
            print(judge_result)

    path_name = f"save_msc/{model1}/infer_rsum_sid5.json"
    r1 = json.load(open(path_name, "r"))
    path_name = f"save_msc/{model2}/infer_sid5.json"
    r2 = json.load(open(path_name, "r"))
    output = []
    new_dial_ids = random.sample(range(0,len(test_dataset)), 500)
    for idx in tqdm(new_dial_ids):
        d = test_dataset[idx]
        dial_id = d['dial_id']
        response1, response2 = r1[dial_id]['prediction'], r2[dial_id]['prediction']
        input_str = prompts['gpt-4']['win_rate'].format_map(
            {'dialog':" ".join(d['all_content'][-10:]), 'persona': d['gt_prev_summary_string'], 'response1': response1, 'response2': response2})
        judge_result = gpt_response_results(input_str, 'gpt-3.5-turbo-1106')
        output.append({
            "dial_id": dial_id,
            "judge_prompt": input_str,
            "evaluation": judge_result})
        
        for pre_idx, pre in enumerate(pre_defined):
            #import pdb; pdb.set_trace()
            if pre in judge_result:
                results[pre_idx] = results[pre_idx] + 1
                break
        
    wfile= os.path.join(args.saving_dir, f"{model1}_{model2}_win_rate.json")
    with open(wfile,"w", encoding='utf-8') as f: 
        f.write(json.dumps(output, ensure_ascii=False, indent=4))
    print(results)
        
    

def run_llm_judge(args, test_dataset):
    file_name = "infer_full_sid5.json"
    path_name = f"{args.saving_dir}/{file_name}"
    result_data = json.load(open(path_name, "r"))
    output_ratings = []
    conditions = {'Fluency': 0, 'Coherence': 0,'Consistency': 0 }

    new_dial_ids = random.sample(range(0,len(test_dataset)), 1000)
    new_data = []
    for dial_id in new_dial_ids:
        new_data.append(test_dataset[dial_id])
        
    for d in tqdm(new_data):
        dial_id = d['dial_id']
        response = result_data[dial_id]['prediction']
        for c in conditions.keys():
            condition_str = prompts['gpt-4'][c]
            judge_prompt = prompts['gpt-4']['single_eval'].format_map(
                {'dialog':" ".join(d['all_content'][-10:]), 'persona': d['gt_prev_summary_string'], 'response': response,'condition':condition_str})
            outputs = gpt_response_results(judge_prompt, 'gpt-4-0314')
            match = re.search(r'\[\[(\d+)\]\]', outputs)
            try:
                rating = int(match.group(1))
            except:
                rating = None
            if rating is not None:
                conditions[c] = conditions[c] + rating
            output_ratings.append({
            "dial_id": dial_id,
            "condition": c,
            "judge_prompt": judge_prompt,
            "evaluation": outputs,
            "rating": rating})
    for c in conditions.keys():
        conditions[c] = conditions[c]/len(new_data) * 1.0

    wfile= os.path.join(args.saving_dir, f"eval_{file_name}")
    with open(wfile,"w", encoding='utf-8') as f: 
        f.write(json.dumps(output_ratings, ensure_ascii=False, indent=4))  

    with open(f'{args.logger_file}', 'a') as f:
        f.write(str(args)+"\n")
        f.write(str(conditions)+"\n")

def load_eval_file(args):
    path_name = f"{args.saving_dir}/eval_question_full_sid5.json"
    result_data = json.load(open(path_name, "r"))
    conditions = {'Coherence': 0,'Consistency': 0, 'Fluency': 0}
    for d in tqdm(result_data):
        if d["condition"] == "Coherence":
            conditions["Coherence"] += d["rating"]
        elif d["condition"] == "Consistency":
            print(d['rating'])
            conditions["Consistency"] += d["rating"]
        elif d["condition"] == "Fluency":
            conditions["Fluency"] += d["rating"]

    for c in conditions.keys():
        conditions[c] = conditions[c]/len(result_data) * 3.0
    print(conditions)
    #import pdb; pdb.set_trace()
    with open(f'{args.logger_file}', 'a') as f:
        f.write(str(args)+"\n")
        f.write(str(conditions)+"\n")

