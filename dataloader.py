import json
from torch.utils.data import DataLoader, Dataset
import random
from functools import partial
from utils.evaluation import compute_f1, compute_f1_sentence
from torch.utils.data import random_split
import copy
import math
from tqdm import tqdm
import torch
import os
count = 0
from dataset import RSumDataset, NerCollate, MSCDataset, RAGDataset, SumDataset

random.seed(42)

carecall_dict = {"user":"User", "system":"Assistant"}
NOPERSONA = '__NO__PERSONA__BEAM__MIN__LEN__20__'
DUMMY_TEXT = '__SILENCE__'
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
PROMPT_TEMPLATE = {
    "summary":  "<s>[INST]<<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\n\
                    ### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n\
                    ### Response:\n Based on the previous text, provide a updated summary:[/INST]",
    "dialog":   "{instruction}\nYou will be provided with a summary containing personality information for both yourself and the user, as well as a dialogue context. Your goal is to respond to the user based on the dialogue context and summary. If there isn't a specific relevant personality trait revelant to user's query, you should still respond naturally and conversationally. The following is a dialogue you need to test: {input}\nResponse to the user within 30 words: ",
    "infer":   "{instruction}\nYou will be provided with dialogue context. Your goal is to respond to the user based on the dialogue context. If there isn't a specific relevant personality trait revelant to user's query, you should still respond naturally and conversationally.\n\
                    The following is a dialogue you need to test: {input}\n\
                    Response to user within 30 words:",
    "summary_ict":"<s>[INST]<<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\n\
                    {instruction}\n\n\
                    I will show you an example. Please return the updated summary of the dialogue:\n{demo_string}\n\n\
                    The following is a dialogue you need to test:\n{input}\n\n\
                    So the updated summary of the dialogue is:[/INST]",
    "dialog_ict": "[INST]<<SYS>>\n{instruction}\n<</SYS>>\n\nYou will be provided with a summary containing personality information for both yourself and the user, as well as a dialogue context. Your goal is to respond to the user based on the dialogue context and summary. If there isn't a specific relevant personality trait revelant to user's query, you should still respond naturally and conversationally. \nI will show you an example as the demonstration: {demo_string}\n\nThe following is a dialogue you need to test: {input}\nResponse to the user within 30 words: [/INST]",
}
IGNORE_INDEX=-100

def count_data(path_name):
    with open(path_name, "r") as f:
        raw_data = [json.loads(line.strip()) for line in f]
    count = 0
    token_num = 0
    for dial in raw_data:
        history = []
        prev_dialogs = dial["previous_dialogs"]
        for prev in prev_dialogs:
            for idx, item in enumerate(prev["dialog"]):
                if idx % 2 == 0:
                    history.append("User: " + item["text"])
                else:
                    history.append("System: " + item["text"])
        for idx, item in enumerate(dial["dialog"]):
            if item["id"] == "Speaker 2":
                token_num += len(" ".join(history).split(" "))
                count += 1
            speaker_id, speaker_text = item["id"], item["text"]
            history.append(f"{speaker_id}: {speaker_text}")

    return token_num / count * 1.00

def collate_fn(batch):
    inputs = [item["input"] for item in batch]
    labels = [item["label"] for item in batch]
    dial_ids = [item["dial_id"] for item in batch]
    return {"input": inputs, "label": labels, "dial_ids":dial_ids}

def load_dataset(args):
    if args.dataset == "msc":
        test_data = read_msc_test_data(args, "test")
    elif args.dataset == "carecall":
        test_path = f'data/carecall/carecall-memory_en_auto_translated.json'
        test_data = read_care_data(args, test_path)
    
    if args.mode == "rsum":
        test_dataset = RSumDataset(test_data, args)
    elif args.mode in ['rag', 'rag_mem']:
        test_dataset = RAGDataset(test_data, args)
    elif args.mode == 'sum':
        test_dataset = SumDataset(test_data, args)
    else:
        test_dataset = MSCDataset(test_data, args)
    
    return test_data, test_dataset


def read_care_data(args, path_name):
    raw_data = json.load(open(path_name))
    data = [[] for i in range(5)]
    ave_num = [0 for i in range(5)]
    resp_num = [0 for i in range(5)]
    for session in raw_data:
        prev_list = []
        all_content = []
        guid_id = session[0]["guid"].split("-")[1]
        for sidx, dial in enumerate(session):
            curr_list = []
            for tidx, turn in enumerate(dial["dialogue"]):
                if turn["role"] == "system":
                    if tidx not in [0, 2]:
                        prev_summary = ""
                        if dial["memory"] != []:
                            prev_summary = "; ".join(dial["memory"]).replace("He/She","The user")
                        dial_item = {
                            "prev_list": prev_list.copy(),
                            'history': prev_list.copy(),
                            "dial_id": guid_id+"_"+str(tidx),
                            'first_turn': tidx == 4,
                            "response": turn["text"],
                            "last_turn": tidx == len(dial["dialogue"]) - 1,
                            "full": " ".join(prev_list.copy()) + " " + " ".join(curr_list),
                            "all_content": all_content.copy(),
                            "window": " ".join(curr_list),
                            "gt_prev_summary_string": prev_summary,
                            "summary": "; ".join(dial["summary"]).replace("He/She","The user")
                        }
                        data[sidx].append(dial_item)

                        ave_num[sidx] = max(ave_num[sidx], len(dial_item['full'].split(" ")))
                text = carecall_dict[turn["role"]] + ": " + turn["text"]
                curr_list.append(text)
                all_content.append(text)

            prev_list.append(" ".join(curr_list))

    '''random.seed(args.random_seed)
    select_data = random.sample(data[args.session_id-1], args.test_num)
    example_data = random.sample(data[args.session_id-1], args.n_shot)
    example = {"window": "", "full": "", "update_response": "", "update_summary": ""}
    for item in example_data:
        window_context, full_context = item["window"], item["full"]
        prev_summary = item["prev_summary"]
        summary = item["summary"]
        response = item["label"]
        example["window"] += f"[Dialogue Context] {window_context} [Response] {response}"
        example[
            "update_summary"] += f"[Previous Memory] {prev_summary} [Dialogue Context] {window_context} [Updated Memory] {summary}"
        example[
            "update_response"] += f"[Previous Memory] {prev_summary} [Dialogue Context] {window_context} [Response] {response}"
        example["full"] += f"[Dialogue Context] {full_context} [Response] {response}"'''

    print(ave_num)
    print(resp_num)
    session_data = data[args.session_id-1]
    if args.mode in ['rag', 'rag_mem']:
        rag_file = f"data/carecall/dpr_topk_3.json"
        session_data = read_rag_data(args, session_data, rag_file)
    return session_data


def read_rag_data(args, dataset, path_name):
    with open(path_name, "r") as f:
        rag_data = json.load(f)
    for idx, dialog in enumerate(rag_data):
        full_context = dialog["full"]
        sentences = []
        topk_index = dialog["rag_result_index"]
        if args.retrieval == 'dpr':
            for item in topk_index:
                sentences.append("User: " + full_context[item[0]]['content'])
                sentences.append("Assistant: " + full_context[item[1]]['content'])
            dataset[idx]["rag"] = " ".join(sentences)
        else:
            dataset[idx]["rag"] = " ".join(dialog['rag_result_context'])
            #import pdb; pdb.set_trace()
    return dataset


def read_msc_data(args, path_name):
    with open(path_name, "r") as f:
        raw_data = [json.loads(line.strip()) for line in f]
    data = []

    #raw_data = random.sample(raw_data, 20)
    for dial in raw_data:
        prev_content = []
        prev_dialogs = dial["previous_dialogs"]
        for prev in prev_dialogs:
            for idx, item in enumerate(prev["dialog"]):
                if idx % 2 == 0:
                    prev_content.append({"role":"user","content":item["text"]})
                else:
                    prev_content.append({"role":"assistant","content":item["text"]})
                    
            if prev_content[-1]["role"] == "user":
                prev_content.pop()

        win_content = []
        for idx, item in enumerate(dial["dialog"]):
            speaker_text = item["text"]
            if idx % 2 == 0:
                speaker_role = "user"
            else:
                speaker_role = "assistant"
                dial_item = {
                        "full": copy.deepcopy(prev_content + win_content),
                        "window": copy.deepcopy(win_content),
                        "dial_id": item["convai2_id"] + "-" + str(idx),
                        "label": item["text"],
                    }
                data.append(dial_item)

            win_content.append({"role": speaker_role, "content":speaker_text})

    print("#Total sample:", len(data))
    if args.do_rag:
        data = rag(data, args.topk)
    
    return data


def prepare_data(args, tokenizer):
    if args.dataset == "msc":
        path_train = f'data/msc_dialog/session_{args.session_id}/train.txt'
        path_dev = f'data/msc_dialog/session_{args.session_id}/valid.txt'
        path_test = f'data/msc_dialog/session_{args.session_id}/test.txt'

        data_train = read_msc_data(args, path_train)
        data_dev = read_msc_data(args, path_dev)
        data_test = read_msc_data(args, path_test)

    elif args.dataset == "carecall":
        path_train = f'data/carecall/carecall-memory_en_auto_translated.json'
        data_train, data_dev, data_test = read_care_data(args, path_test)

    train_dataset = MSCDataset(data_train)
    dev_dataset = MSCDataset(data_dev)
    test_dataset = MSCDataset(data_test)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                              collate_fn=partial(train_collate_fn, tokenizer=tokenizer, args=args))
    dev_loader = DataLoader(dev_dataset, batch_size=args.dev_batch_size, shuffle=False,
                            collate_fn=partial(train_collate_fn, tokenizer=tokenizer, args=args))
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                             collate_fn=partial(train_collate_fn, tokenizer=tokenizer, args=args))

    return train_loader, dev_loader, test_loader

def prepare_sum_data(args):

    path_train = f'data/msc_personasummary/session_{args.session_id}/train.txt'
    path_dev = f'data/msc_personasummary/session_{args.session_id}/valid.txt'
    path_test = f'data/msc_personasummary/session_{args.session_id}/test.txt'

    data_train = read_sum_data(args, path_train, "train")
    data_dev = read_sum_data(args, path_dev, "dev")
    data_test = read_sum_data(args, path_test, "test")

    train_dataset = SumDataset(data_train)
    dev_dataset = SumDataset(data_dev, split='dev')
    test_dataset = SumDataset(data_test, split='test')

    return train_dataset, dev_dataset, test_dataset


def sum_collate_fn(batch, args, demo=None):
    def constr_input(summary, context, operation=None, label=None):
        if operation == "dialog_ict":
            return f"Summary: {summary}\nDialogue context: {context}\nResponse to the user within 30 words: assistant: {label}"
        if operation == "summary_ict":
            return f"Summary: {summary}\nDialogue context: {context}\nSo the updated summary is: Summary: {label}"
    inputs = []
    labels = []
    dial_ids = []
    #import pdb; pdb.set_trace()
    prompt = PROMPT_TEMPLATE[args.operation]
    summary_name = f'{args.summary_type}_prev_summary_string'
    if demo is not None:
        demo_string = constr_input(demo[summary_name], demo[args.mode], args.operation, demo['label'])
    for example in batch:
        output = example["label"]
        if args.operation == 'infer':
            input_string = f"Dialogue context: {example[args.mode]}"
        else:
            input_string = f"Summary: {example[summary_name]}\nDialogue context: {example[args.mode]}"
        if "ict" in args.operation:
            source = prompt.format_map({'instruction': example["instruction"],"input": input_string, "demo_string":demo_string})
        else:
            source = prompt.format_map({'instruction': example["instruction"],"input": input_string})
        inputs.append(source)
        labels.append(output)
        dial_ids.append(example["dial_id"])

    return {"input": inputs, "label": labels, "dial_ids":dial_ids}

def prepare_rsum_data(args):
    data_test = read_rsum_data(args, "test")
    data_valid = read_rsum_data(args, "valid")
    data_demo = random.sample(data_valid, 1)[0]
    data_demo['label'] = data_demo['response']
    test_dataset = RSumDataset(data_test, args)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, 
                             shuffle=False, collate_fn=partial(sum_collate_fn, args=args, demo=data_demo), num_workers=0)
    return test_loader

def read_msc_test_data(args, dtype):
    def merge_personas(init_personas, add_personas):
        
        if add_personas[0] == [] and add_personas[1] == []:
            return init_personas

        new_personas = [] 
        for i in range(2):
            personas_i = init_personas[i].copy()
            for ip in add_personas[i]:
                found = False
                for j, op in enumerate(personas_i):
                    if compute_f1_sentence(ip, op) > 0.4:
                        personas_i[j] = ' '.join([op, ip])
                        found = True
                        break
                if not found:
                    personas_i.append(ip)

            new_personas.append(personas_i.copy())
        return new_personas
    
    def load_summary_file(args):
        file_path = os.path.join("save_msc/gpt-3.5-old", f"infer_sum_sid5.json")
        if not os.path.exists(file_path): 
            return None
        with open(file_path, 'r') as f:
            result = json.load(f)
        return result
    

    def get_person_string(personas_list):
        user_persona = " ".join(personas_list[0])
        assistant_persona = " ".join(personas_list[1])
        dict_persona = "User: "+ user_persona+" Assistant: "+assistant_persona
        return dict_persona
    
    path_name = f'data/msc_dialogue/session_{args.session_id}/{dtype}.txt'
    with open(path_name, "r") as f:
        raw_data = [json.loads(line.strip()) for line in f]

    data = []
    persona_list = {}
    sum_data = []
    prev_summary_set = load_summary_file(args)

    for dialog_dict in tqdm(raw_data):
        dial_id = dialog_dict["metadata"]["initial_data_id"] 
        pred_prev_summary = [""]
        if prev_summary_set is not None and dial_id in prev_summary_set.keys():
            #import pdb; pdb.set_trace()
            pred_prev_summary = prev_summary_set[dial_id]
    
        #prev_sum = [{"role":"user", "content":prev_summary["user persona"]}]
        #prev_sum.append({"role":"assistant","content":prev_summary["assistant persona"]})
        prev_dialogs = dialog_dict["previous_dialogs"]

        init_dialog =  dialog_dict['previous_dialogs'][0]
        gt_prev_summary = init_dialog['personas']
        if len(dialog_dict['previous_dialogs']) > 1:
            for prev_dialog in dialog_dict['previous_dialogs'][1:]:
                gt_prev_summary = merge_personas(gt_prev_summary, prev_dialog['personas'])

        history = []
        prev_content_list = []
        all_content = []
        for prev_dialog in prev_dialogs:
            prev_content = []
            for idx, item in enumerate(prev_dialog["dialog"]):
                prefix = "User" if idx % 2 == 0 else "Assistant"
                prev_content.append(f'{prefix}: {item["text"]}')
                prev_content_list.append(f'{prefix}: {item["text"]}')
                all_content.append(f'{prefix}: {item["text"]}')
            history.append(" ".join(prev_content))

        win_content = []      

        sum_action = {
            "history": history,
            'pred_prev_summary': [],
            'dial_id': dial_id,
            'gt_prev_summary': gt_prev_summary,
        }      
        sum_data.append(sum_action)
        for tidx, turn in enumerate(dialog_dict["dialog"]):
            prefix = "User" if tidx % 2 == 0 else "Assistant"
            if prefix == "Assistant":
                action = {
                    "full": " ".join(history + win_content),
                    "prev_content": prev_content_list.copy(),
                    'win_content': win_content.copy(),
                    'all_content': all_content.copy(),
                    "history": history, 
                    "window": " ".join(win_content),
                    'response': turn["text"],
                    'pred_prev_summary': pred_prev_summary,
                    'gt_prev_summary': gt_prev_summary,
                    'gt_prev_summary_string': get_person_string(gt_prev_summary),
                    'dial_id': dial_id + "-" + str(tidx),
                }
                data.append(action)
                
            win_content.append(f'{prefix}: {turn["text"]}')
            all_content.append(f'{prefix}: {turn["text"]}')

    if args.mode in ['rag', 'rag_mem']:
        rag_file = f"data/msc_dialogue/session_{args.session_id}/dpr_topk_3.json"
        data = read_rag_data(args, data, rag_file)
    #import pdb; pdb.set_trace()
    if args.mode == 'sum':
        return sum_data
    else:
        return data[:1000]

def create_excel(data):
    import openpyxl

    raw_data = random.sample(data, 100)
    # 创建一个新的Excel工作簿
    workbook = openpyxl.Workbook()

    # 获取默认的工作表
    sheet = workbook.active

    # 写入数据
    sheet['A1'] = 'Test_ID'
    sheet['B1'] = 'Context'
    sheet['C1'] = 'Persona'
    sheet['D1'] = 'R1'
    sheet['E1'] = 'R2'
    sheet['F1'] = 'R3'
    sheet['G1'] = 'R4'
    
    mode_list = ["chatgpt", "dialog","infer_rag","infer_window"]

    fileHandler = open(f"save_msc/gpt-3.5-old/infer_rsum_sid5.json",  "r")
    infers = json.load(fileHandler)
    fileHandler = open(f"save50/llama2-13b-chat/msc_dialog_sid5.json",  "r")
    ours = json.load(fileHandler)
    fileHandler = open(f"save50/llama2-13b-chat/msc_infer_rag_sid5.json",  "r")
    rags = json.load(fileHandler)
    fileHandler = open(f"save50/llama2-13b-chat/msc_infer_window_sid5.json",  "r")
    windows = json.load(fileHandler)

    for idx, item in enumerate(raw_data):
        label = item["response"]
        persona_list = []
        for a_p in item['gt_prev_summary'][1]:
            if compute_f1_sentence(label, a_p) > 0.1:
                print(compute_f1_sentence(label, a_p))
                persona_list.append(a_p)
        id = item["dial_id"]
        sheet_list = [item["dial_id"], item['window'], '\n'.join(persona_list), infers[id]['prediction'],
                      ours[id]['prediction'],rags[id]['prediction'],windows[id]['prediction']]

        sheet.append(sheet_list.copy())
        if (idx+1) % 20 == 0:
            workbook.save(f'example{str(idx+1)}.xlsx')
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            sheet['A1'] = 'Test_ID'
            sheet['B1'] = 'Context'
            sheet['C1'] = 'Persona'
            sheet['D1'] = 'R1'
            sheet['E1'] = 'R2'
            sheet['F1'] = 'R3'
            sheet['G1'] = 'R4'


def prepare_test_sum_data(args):
    data_test = read_test_sum_data(args, "test")
    data_valid = read_test_sum_data(args, "valid")
    data_demo = random.sample(data_valid, 1)
    test_dataset = SumDataset(data_test, args.local_rank, split='test')
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, 
                             shuffle=False, collate_fn=partial(sum_collate_fn, args=args, demo=data_demo), num_workers=0)
    return test_loader

def read_sum_test_data(args, dtype):
    def merge_personas(init_personas, add_personas):
        if add_personas[0] == [] and add_personas[1] == []:
            return init_personas

        new_personas = [] 
        for i in range(2):
            personas_i = init_personas[i].copy()
            for ip in add_personas[i]:
                found = False
                for j, op in enumerate(personas_i):
                    if compute_f1_sentence(ip, op) > 0.4:
                        personas_i[j] = ' '.join([op, ip])
                        found = True
                        break
                if not found:
                    personas_i.append(ip)

            new_personas.append(personas_i.copy())
        return new_personas
    
    def load_summary_file(args):
        file_path = os.path.join(args.saving_dir, f"{args.dataset}_summarize_sid{args.session_id-1}.json")
        if not os.path.exists(file_path): 
            return None
        with open(file_path, 'r') as f:
            result = json.load(f)
        return result
    
    def get_person_string(personas_list):
        user_persona = " ".join(personas_list[0])
        assistant_persona = " ".join(personas_list[1])
        dict_persona = {"user persona":user_persona, "assistant persona":assistant_persona}
        return str(dict_persona)

    path_name = f'data/msc_dialogue/session_{args.session_id}/{dtype}.txt'
    with open(path_name, "r") as f:
        raw_data = [json.loads(line.strip()) for line in f]

    data = []
    prev_summary_set = load_summary_file(args)
    #raw_data = raw_data[:10]
    for dialog_dict in tqdm(raw_data):
        prev_dialog = dialog_dict['previous_dialogs'][-1]["dialog"]
        context = []
        for prev_dialog in dialog_dict['previous_dialogs']:
            last_dialog = prev_dialog['dialog']
            dialog_texts = []
            for i in range(0, len(last_dialog)):
                prefix = "User: " if i % 2 == 0 else "Assistant: "
                dialog_texts.append(f'{prefix}{last_dialog[i]["text"]}')
            context.append(" ".join(dialog_texts))

        init_dialog =  dialog_dict['previous_dialogs'][0]
        personas = init_dialog['personas']
        if len(dialog_dict['previous_dialogs']) > 1:
            for prev_dialog in dialog_dict['previous_dialogs'][1:]:
                personas = merge_personas(personas, prev_dialog['personas'])

        dial_id = dialog_dict["metadata"]["initial_data_id"]
        prev_summary = {"user persona":"", "assistant persona":""}
        if prev_summary_set is not None and dial_id in prev_summary_set.keys():
            prev_summary = prev_summary_set[dial_id]["prediction"]
        action = {
                'context': '\n'.join(dialog_texts),
                'label': personas,
                'label_string': get_person_string(personas),
                'prev_summary': prev_summary,
                'prev_summary_string': str(prev_summary),
                'dial_id': dial_id,
            }
        data.append(action)
        
    return data

def read_sum_data(args, path_name, data_type):
    def merge_personas(init_personas, add_personas):
        new_personas = [] 
        if add_personas[0] == [] and add_personas[1] == []:
            return init_personas

        for i in range(2):
            personas_i = init_personas[i].copy()
            for ip in add_personas[i]:
                found = False
                for j, op in enumerate(personas_i):
                    if compute_f1_sentence(ip, op) > 0.5:
                        personas_i[j] = ' '.join([op, ip])
                        found = True
                        break
                if not found:
                    personas_i.append(ip)

            new_personas.append(personas_i.copy())
        return new_personas
    
    def get_person_string(personas_list):
        user_persona = ' '.join(personas_list[0])
        assistant_persona = ' '.join(personas_list[1])
        dict_persona = {"user persona":user_persona, "assistant persona":assistant_persona}
        return str(dict_persona)

    def get_person_string(personas_list):
        user_persona = " ".join(personas_list[0])
        assistant_persona = " ".join(personas_list[1])
        dict_persona = {"user persona":user_persona, "assistant persona":assistant_persona}
        return str(dict_persona)

    with open(path_name, "r") as f:
        raw_data = [json.loads(line.strip()) for line in f]

    data = []
    negative_data = []
    #raw_data = random.sample(raw_data, 20)
    for dialog_dict in tqdm(raw_data):
        current_episode = dialog_dict['dialog']
        init_personachat = dialog_dict['init_personachat']
        init_personas = init_personachat['init_personas']
        init_persona_string = get_person_string(init_personas)
        for end_idx in range(len(current_episode)):
            if args.summary_num_turns > 0:
                start_index = max(0, end_idx - args.summary_num_turns + 1)
            else:
                start_index = 0

            add_personas = [[],[]]
            end_line_persona = (
                current_episode[end_idx]['persona_text']
                if 'persona_text' in current_episode[end_idx]
                else NOPERSONA
                )
            
            idx1, idx2 = end_idx, end_idx - 1
            r1, r2 = idx1 % 2 , idx2 % 2 
            add_personas[r1] = current_episode[idx1]['agg_persona_list']
            if idx2 >= 0 :
                add_personas[r2] = current_episode[idx2]['agg_persona_list']

            end_personas = merge_personas(init_personas, add_personas)
            end_persona_string = get_person_string(end_personas)

            dialog_texts = []
            for i in range(start_index, end_idx + 1):
                prefix = "partner: " if i % 2 == 0 else "you: "
                dialog_texts.append(f'{prefix}{current_episode[i]["text"]}')

            action = {
                'context': ' '.join(dialog_texts),
                'labels': end_persona_string,
                'initial_data_id': dialog_dict['initial_data_id'],
                'init_personas': init_persona_string,
                'utt_idx': end_idx,
                'speaker_idx': end_idx % 2,
                'session_id': args.session_id,
            }
            if end_line_persona == NOPERSONA:
                negative_data.append(action)
            else:
                data.append(action)

    if data_type == 'train':
        size_to_sample = math.ceil(args.nopersona_subsampling_weight * len(negative_data))
        data.extend(random.sample(negative_data, size_to_sample))
    else:
        data.extend(negative_data)
    
    print(f"#Total number:{len(data)}")
    return data
