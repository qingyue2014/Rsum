import re
import json
import time
import os
import tiktoken
from tqdm import tqdm
from random import sample
from openai import OpenAI
import openai
import httpx
openai_modelid = 'gpt-3.5-turbo-0301' 
prompt_path = 'memo_chat/prompts.json'
#encoding = tiktoken.encoding_for_model(openai_modelid)

q_pre = ""
qa_link = ""
MaxLen = 2048
TarLen = 512
TaskTarLen = {
    "chatting_dialogsum": MaxLen,
    "chatting_alpacagpt4": MaxLen,
    "writing_topiocqa": TarLen // 2,
    "writing_dialogsum": TarLen,
    "retrieval_dialogsum": 32,
    "retrieval_topiocqa": 32
}

client = OpenAI(
    api_key="xxxxxx"
)

prompts = json.load(open(prompt_path, "r"))

def normalize_model_outputs(model_text):
    extracted_elements = [re.sub(r'\s+', ' ', mt.replace('"', '').replace("'", "")) for mt in re.findall(r"'[^']*'|\"[^\"]*\"|\d+", model_text)]
    model_outputs = []
    ti = 0
    while ti + 7 < len(extracted_elements):
        if extracted_elements[ti] == "topic" and extracted_elements[ti + 2] == "summary" and extracted_elements[ti + 4] == "start" and extracted_elements[ti + 6] == "end":
            try:
                model_outputs.append({"topic": extracted_elements[ti + 1], "summary": extracted_elements[ti + 3], "start": int(extracted_elements[ti + 5]), "end": int(extracted_elements[ti + 7])})
            except:
                pass
        ti += 1
    return model_outputs

def normalize_chatting_outputs(model_outputs):
    def white_space_fix(text):
        lines = text.split("\n")
        result = []
        for line in lines:
            result.append(' '.join(line.split()))
        output = '\n'.join(result)
        return output
    return white_space_fix(model_outputs)

def gen_model_output(input_qs, task_type):
    target_len = TaskTarLen[task_type]
    messages = [{"role": "user", "content": input_qs}]
    for i in range(100):
        try:
            chat = client.chat.completions.create(
                model=openai_modelid, messages=messages, max_tokens=target_len, temperature=0)
            break
        except:
            time.sleep(5)
    model_outputs = chat.choices[0].message.content
    return model_outputs

def run_summary(history, memo, bot_thinking):
    system_insturction = prompts["writing_dialogsum"]["system"]
    task_instruction = prompts["writing_dialogsum"]["instruction"]
    history_log = "\n\n```\nTask Conversation:\n" + "\n".join(["(line {}) {}".format(h_i + 1, h.replace("\n", " ")) for h_i, h in enumerate(history["Recent Dialogs"][2:])])
    qs = q_pre + system_insturction.replace("LINE", str(len(history["Recent Dialogs"]) - 2)) + history_log + "\n```" + task_instruction.replace("LINE", str(len(history["Recent Dialogs"]) - 2)) + qa_link

    sum_history = gen_model_output(qs, "writing_dialogsum")
    sum_history = normalize_model_outputs(sum_history)

    for s in sum_history:
        memo[s["topic"]] = memo.get(s["topic"], []) + [{"summary": s["summary"], "dialogs": history["Recent Dialogs"][2:][(s["start"] - 1):s["end"]]}]
    if len(sum_history) == 0:
        si_0, si_1 = sample(list(range(len(history["Recent Dialogs"][2:]))), 2)
        memo["NOTO"].append({"summary": "Partial dialogs about: {} or {}.".format(history["Recent Dialogs"][2:][si_0], history["Recent Dialogs"][2:][si_1]), "dialogs": history["Recent Dialogs"][2:]})
    history["Recent Dialogs"] = history["Recent Dialogs"][-2:]
    bot_thinking["summarization"] = {"input": qs, "output": sum_history}
    return history, memo, bot_thinking

def run_retrieval(history, memo, bot_thinking):
    topics = []
    for k, v in memo.items():
        for vv in v:
            topics.append((k, vv["summary"], vv["dialogs"]))
    system_insturction = prompts["retrieval"]["system"]
    task_instruction = prompts["retrieval"]["instruction"]
    task_case = "```\nQuery Sentence:\n" + history["User Input"][6:] + "\nTopic Options:\n" + \
                "\n".join(["({}) {}".format(v_i + 1, v[0] + ". " + v[1]) for v_i, v in enumerate(topics)]) + "\n```"
    qs = q_pre + system_insturction.replace("OPTION", str(len(topics))) + task_case + task_instruction.replace("OPTION", str(len(topics))) + qa_link
    # print("-" * 20 + "retrieving" + "-" * 20)
    # print(qs)
    # print("-" * 20 + "retrieving" + "-" * 20)
    outputs = gen_model_output(qs, "retrieval_dialogsum")
    # print("-" * 20 + "retrieval" + "-" * 20)
    # print(outputs)
    # print("-" * 20 + "retrieval" + "-" * 20)
    outputs = outputs.split("#")
    chosen_topics = []
    for output in outputs:
        try:
            index_ = int(output) - 1
        except:
            continue
        if index_ < len(topics) and "NOTO" not in topics[index_]:
            chosen_topics.append(topics[index_])
    if len(chosen_topics) > 0:
        history["Related Topics"] = [ct[0] for ct in chosen_topics]
        history["Related Summaries"] = [ct[1] for ct in chosen_topics]
        history["Related Dialogs"] = [" ### ".join(ct[2]) for ct in chosen_topics]
    else:
        history["Related Topics"] = []
        history["Related Summaries"] = []
        history["Related Dialogs"] = []
    bot_thinking["retrieval"] = {"input": qs, "output": outputs}
    return history, bot_thinking

def run_memochat(args, test_dataset):
    pred_dicts = {}
    history = {
            "Recent Dialogs": [], 
            "Related Topics": [], 
            "Related Summaries": [], 
            "Related Dialogs": [], 
            "User Input": "",
            }
    memo = {"NOTO": [{"summary": "None of the others.", "dialogs": []}]}
    for d in tqdm(test_dataset):
        new_d = d
        if d['first_turn']:
            history = {
            "Recent Dialogs": [], 
            "Related Topics": [], 
            "Related Summaries": [], 
            "Related Dialogs": [], 
            "User Input": "",
            }
            memo = {"NOTO": [{"summary": "None of the others.", "dialogs": []}]}
        dialogue_context = []
        
        for i in range(int(len(new_d["all_content"])/2)):
            dialogue_context.append(new_d["all_content"][2*i] + " " + new_d["all_content"][2*i+1])
        bot_thinking = {"retrieval": "", "summarization": ""}
        l_i = 0
        if d['first_turn']:
            while l_i < len(dialogue_context)-1:
                history["Recent Dialogs"] += [dialogue_context[l_i], dialogue_context[l_i + 1]]
                # create summary if recent dialogs exceed threshold
                if len(" ### ".join(history["Recent Dialogs"]).split(" ")) > (MaxLen // 2) or len(history["Recent Dialogs"]) >= 10:
                    history, memo, bot_thinking = run_summary(history, memo, bot_thinking)
                l_i = l_i + 2

                # retrieve most related topics for every new user input
        history["User Input"] = dialogue_context[-1]
        if len(memo.keys()) > 1:
            history, bot_thinking = run_retrieval(history, memo, bot_thinking)
                
        # generate bot response
        system_insturction = prompts["chatting"]["system"]
        task_instruction = prompts["chatting"]["instruction"]
        task_case = "```\nRelated Evidences:\n" + "\n".join(["({}) {}".format(r_tsd_i + 1, {
                                "Related Topics": history["Related Topics"][r_tsd_i], 
                                "Related Summaries": history["Related Summaries"][r_tsd_i], 
                                "Related Dialogs": history["Related Dialogs"][r_tsd_i]
                            }) for r_tsd_i in range(len(history["Related Topics"]))]) + "\n\nRecent Dialogs:\n" + \
                            " ### ".join([hrd.replace("\n", " ") for hrd in history["Recent Dialogs"]]) + "\n```\n\nUser Input:\n" + history["User Input"] + " ### bot: "
        qs = q_pre + system_insturction + task_case + task_instruction + qa_link
        outputs = gen_model_output(qs, "chatting_dialogsum")
        outputs = normalize_chatting_outputs(outputs)
        pred_dicts[new_d['dial_id']] = {'prediction': outputs, 'label': new_d['response']}
    
    wfile= os.path.join(args.saving_dir, f"{args.operation}_sid{args.session_id}.json")
    with open(wfile,"w", encoding='utf-8') as f: 
        f.write(json.dumps(pred_dicts, ensure_ascii=False, indent=4))  


